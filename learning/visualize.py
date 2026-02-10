from brax.training.agents.ppo import networks as ppo_networks
import functools
from mujoco_playground import registry
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
from mujoco_playground.config import locomotion_params

import time
import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer
import pickle

from worldModel.train_world_model import create_train_state, train_step, ALL_ENVS
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

from collections import deque
from scipy.stats import ks_2samp
from river.drift import ADWIN

IMPL = "jax"

class KSDriftDetector:
    def __init__(self, total_size=1000, window_size=250, adwin_delta=1e-3):
        """
        KS-based drift detector with ADWIN on the KS statistic.

        Args:
            total_size (int): Total buffer length (Reference + Window).
            window_size (int): Size of the current window.
            adwin_delta (float): ADWIN confidence parameter.
                                 Smaller = fewer false alarms.
        """
        self.buffer = deque(maxlen=total_size)
        self.stat_values = []
        self.window_size = window_size
        
        # Minimum samples needed before we start testing
        # We need at least full window + some reference
        self.min_samples = window_size * 2

        # ADWIN monitors the statistic stream
        self.adwin = ADWIN(delta=adwin_delta)

    def update(self, error_val):
        """
        Input: error_val (float)
        Returns: is_drift (bool), statistic (float)
        """
        self.buffer.append(error_val)

        # Check if we have enough data
        if len(self.buffer) < self.min_samples:
            self.stat_values.append(0)
            return False, 0.

        data = list(self.buffer)
        
        # Reference: Everything EXCEPT the last N elements
        # Window: The last N elements
        reference_data = data[:-self.window_size]
        window_data = data[-self.window_size:]

        # Run KS Test
        statistic, _ = ks_2samp(reference_data, window_data)
        alpha = 0.1
        statistic = alpha * statistic + (1-alpha) * self.stat_values[-1]

        self.stat_values.append(statistic)
        
        self.adwin.update(statistic)
        is_drift = self.adwin.drift_detected

        if is_drift:
            self.reset(data)

        return is_drift, statistic
    
    # The reference data at the point of a domain change detection is filled with the previous domain's prediction errors. 
    # This is stale data as now we are only concerned about the data from the new domain the robot is in.
    # In this method, the reference data is cleared and the adwin detector is also reset
    def reset(self, data):
        self.buffer.clear()
        self.buffer.extend(data[-self.window_size:])
        self.adwin._reset()
    
def load_env(env_name, impl):
    env_cfg = registry.get_default_config(env_name)
    env_cfg["impl"] = impl

    return registry.load(env_name, config=env_cfg)

# Work in progress controller for the robot
class RobotController:
    def __init__(self, obs_shape, act_shape, env_name=None, checkpoint_path=None, jit_inference=None):
        self.wms = []
        self.loadWorldModels(env_name)

        if jit_inference is None:
            self.loadPolicy(env_name, checkpoint_path, obs_shape, act_shape)
        else:
            self.inference = jit_inference
        
        # JIT compile the gradient descent logic
        self.fast_update = jax.jit(train_step)
        
        # History for domain detection
        self.errors = []
        self.drift_indices = []

        # total_size=1000: Takes 20 seconds to fill the queue
        # window_size=250: Detect changes based on the last 5 seconds of data
        self.detector = KSDriftDetector(total_size=1000, window_size=250, adwin_delta=1e-2)
        self.buffer = deque(maxlen=100)

    def loadWorldModels(self, initial_env):
        for env_name in ALL_ENVS:
            root = f"world_models/{env_name}/"

            # Load Pre-trained Weights
            with open(root + "world_model_best.pkl", 'rb') as f:
                params = pickle.load(f)
            with open(root + "normalization_stats.pkl", 'rb') as f:
                stats = pickle.load(f)
                
            sensor_dim = stats['obs_mean'].shape[0]
            action_dim = stats['act_mean'].shape[0]

            # Initialize training state (Optimizer)
            rng = jax.random.PRNGKey(0)
            wm_state = create_train_state(rng, learning_rate=1e-5, sensor_dim=sensor_dim, action_dim=action_dim) # Low LR for stability
            self.wms.append((env_name, wm_state.replace(params=params), stats))

            # The active world model is the one corresponding to the environment we launch the robot in
            if env_name == initial_env:
                print(f"Starting up with {env_name} world model")
                self.active_wm = self.wms[-1]

    def loadPolicy(self, env_name, checkpoint_path, obs_shape, act_shape):
        # Load network topology
        ppo_params = locomotion_params.brax_ppo_config(env_name, IMPL)

        normalize = lambda x, y: x
        if ppo_params.normalize_observations:
            normalize = running_statistics.normalize

        network_fn = (
            ppo_networks.make_ppo_networks
        )
        if hasattr(ppo_params, "network_factory"):
            network_factory = functools.partial(
                network_fn, **ppo_params.network_factory
            )
        else:
            network_factory = network_fn
        
        ppo_network = network_factory(
            obs_shape, act_shape, preprocess_observations_fn=normalize
        )

        make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

        print(f"Loading weights from: {checkpoint_path}")
        params = checkpoint.load(checkpoint_path)

        inference_fn = make_inference_fn(params, deterministic=True)
        self.inference = jax.jit(inference_fn)
        
    def getPredictionWM(self, wm, stats, obs, action):
        # Predict what should have happened
        norm_obs = (obs - stats['obs_mean']) / stats['obs_std']
        norm_act = (action - stats['act_mean']) / stats['act_std']

        norm_delta = wm.apply_fn(
            {'params': wm.params}, norm_obs, norm_act
        )

        # Denormalize to check error in real units (e.g. degrees per second)
        return obs + (norm_delta * stats['delta_std']) + stats['delta_mean']
    
    def meanErrorWM(self, wm, stats):
        batch_obs, batch_act, batch_next = zip(*self.buffer)
        
        # Stack into JAX arrays (Shape: [Batch_Size, Dim])
        obs_arr = jp.stack(batch_obs)
        act_arr = jp.stack(batch_act)
        next_arr = jp.stack(batch_next)
        
        return jp.mean((self.getPredictionWM(wm, stats, obs_arr, act_arr) - next_arr) ** 2)

    def control_loop(self, obs, action, next_obs):
        self.buffer.append((obs, action, next_obs))

        active_name, wm, stats = self.active_wm
        pred_next_obs = self.getPredictionWM(wm, stats, obs, action)
        
        # Calculate error
        error = jp.mean((pred_next_obs - next_obs) ** 2)
        self.errors.append(error)
        
        # Update detector
        is_drift, statistic = self.detector.update(error)

        step = len(self.detector.stat_values)

        if is_drift:
            idx = step - 1
            self.drift_indices.append(idx)
            print(f"!!! DOMAIN CHANGE DETECTED !!! statistic={statistic:.2e}.")
            
            best = min([(self.meanErrorWM(wm, stats), name, wm, stats) for name, wm, stats in self.wms if name != active_name])
            _, best_name, best_wm, best_stats = best

            self.active_wm = (best_name, best_wm, best_stats)
            print(f"Continuing with {best_name}.")

        if is_drift and step > self.detector.min_samples:
            #plt.hist(self.errors, bins=25, range=(0, 2))
            #plt.title("Error distribution on flat ground (Zoomed 0-2)")
            plt.plot(self.detector.stat_values, label="KS statistic")

            plt.vlines(self.drift_indices,
                ymin=min(self.detector.stat_values),
                ymax=max(self.detector.stat_values),
                color="red", alpha=0.6, label='Drift detection')

            plt.xlabel("Time step")
            plt.title("KS-ADWIN concept drift detector history")
            plt.legend()
            plt.tight_layout()
            plt.show()

def interactive_visualization(env, controller=None, resetNum=-1, jit_inference=None):
    """
    Opens an interactive MuJoCo viewer for a JAX-based environment.
    
    Args:
        env: The mujoco_playground environment (unwrapped or wrapped).
        params: The trained policy parameters.
        inference_fn: The function make_inference_fn(params, deterministic=True).
    """

    if controller is None:
        obs_shape, act_shape = env.observation_size, env.action_size
        controller = RobotController(obs_shape, act_shape, jit_inference=jit_inference)

    # Get the underlying standard MuJoCo model for the viewer
    if hasattr(env, 'mj_model'):
        model = env.mj_model
    else:
        # Fallback if environment is wrapped, try to access via unwrapped
        model = env.unwrapped.mj_model
        
    data = mujoco.MjData(model)
    rng = jax.random.PRNGKey(0)
    
    if not hasattr(env, 'jit_reset'):
        env.jit_reset = jax.jit(env.reset)
        env.jit_step = jax.jit(env.step)

    # Initialize the Simulation State
    rng, key1 = jax.random.split(rng)
    state = env.jit_reset(rng)
    
    reset_timer = 0
    control_dt = getattr(env, 'dt', model.opt.timestep)
    print(f"Simulation running at control DT: {control_dt:.4f}s")

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Initialize viewer camera if needed
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.5]
        
        step_start = time.time()
        
        while viewer.is_running():
            # Reset after 10 seconds
            if reset_timer >= 10:
                #print("Resetting")
                rng, key1 = jax.random.split(rng)
                state = env.jit_reset(rng)
                reset_timer = 0
                resetNum -= 1

                if resetNum == 0:
                    break

            # Instruct robot to go always forwards
            state.info["command"] = jp.array([1., 0., 0.])

            step_start = time.time()
            rng, act_rng = jax.random.split(rng)
            
            # Run Policy
            action = controller.inference(state.obs, act_rng)[0]

            # Step environment
            prev_obs = state.obs["wm_state"]
            state = env.jit_step(state, action)
            controller.control_loop(prev_obs, action, state.obs["wm_state"])

            # Ensure computation is done before we try to read it
            state.data.qpos.block_until_ready()
            
            # Extract qpos/qvel from JAX to Numpy (CPU)
            # Copy to the viewer's data structure
            data.qpos[:] = np.array(state.data.qpos)
            data.qvel[:] = np.array(state.data.qvel)

            # Forward kinematics (compute world positions of geoms based on qpos)
            mujoco.mj_forward(model, data)

            # Sync the viewer
            viewer.sync()

            reset_timer += control_dt
            elapsed = time.time() - step_start
            # Sleep only the remaining time to match real-time
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

def main():
    '''
    Loads the omni-directional policy network and environment for the Go2 Stroll environment.
    Then, it simulates it in the Mujoco interactive viewer using JAX.
    '''

    env_name = "Go2StrollFlatTerrain"
    checkpoint_path = "/home/marcos/Escritorio/mujoco_playground/logs/Go2StrollFlatTerrain-20260122-183735/checkpoints/000200540160"
    #checkpoint_path = "/home/marcos/Escritorio/mujoco_playground/logs/Go2StrollRoughTerrain-20260208-001847/checkpoints/000200540160"

    env = load_env(env_name, IMPL)
    env.jit_reset = jax.jit(env.reset)
    env.jit_step = jax.jit(env.step)

    rough_env = load_env("Go2StrollRoughTerrain", IMPL)
    rough_env.jit_reset = jax.jit(rough_env.reset)
    rough_env.jit_step = jax.jit(rough_env.step)

    obs_shape, act_shape = env.observation_size, env.action_size

    #env_name = "Go2StrollRoughTerrain"
    controller = RobotController(obs_shape, act_shape, env_name=env_name, checkpoint_path=checkpoint_path)

    interactive_visualization(env, controller=controller, resetNum=2)
    interactive_visualization(rough_env, controller=controller, resetNum=2)
    interactive_visualization(env, controller=controller, resetNum=2)
    interactive_visualization(rough_env, controller=controller, resetNum=2)

if __name__ == "__main__":
    main()