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

from worldModel.train_world_model import create_train_state, train_step, MODEL_SAVE_PATH, STATS_SAVE_PATH
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

from collections import deque
from scipy.stats import ks_2samp

class KSDriftDetector:
    def __init__(self, total_size=1000, window_size=250, p_threshold=1e-4):
        """
        A sliding window drift detector.

        Args:
            total_size (int): Total buffer length (Reference + Window).
                              e.g., 1000 steps = 20 seconds at 50Hz.
            window_size (int): The size of the "Current" window to test.
                               The 'Reference' size becomes (total_size - window_size).
            p_threshold (float): Sensitivity. Lower = less sensitive.
        """
        self.buffer = deque(maxlen=total_size)
        self.window_size = window_size
        self.p_threshold = p_threshold
        
        # Minimum samples needed before we start testing
        # We need at least full window + some reference
        self.min_samples = window_size * 2 

    def update(self, error_val):
        """
        Input: error_val (float)
        Returns: is_drift (bool), p_value (float)
        """
        self.buffer.append(error_val)

        # Check if we have enough data
        if len(self.buffer) < self.min_samples:
            return False, 1.0

        data = list(self.buffer)
        
        # Reference: Everything EXCEPT the last N elements
        # Window: The last N elements
        reference_data = data[:-self.window_size]
        window_data = data[-self.window_size:]

        # Run KS Test
        _, pvalue = ks_2samp(reference_data, window_data)

        # Detection logic
        # - pvalue < threshold: The distributions look different
        # - mean(window) > mean(ref): The error got WORSE (we ignore if it gets better)
        is_drift = (pvalue < self.p_threshold) and (np.mean(window_data) > np.mean(reference_data))

        return is_drift, pvalue
    
def load_env(env_name, impl):
    env_cfg = registry.get_default_config(env_name)
    env_cfg["impl"] = impl

    return registry.load(env_name, config=env_cfg)

# Work in progress controller for the robot
# TODO: integrate the policy loading logic into the controller
class RobotController:
    def __init__(self):
        # Load Pre-trained Weights
        with open(MODEL_SAVE_PATH, 'rb') as f:
            params = pickle.load(f)
        with open(STATS_SAVE_PATH, 'rb') as f:
            self.stats = pickle.load(f)
            
        sensor_dim = self.stats['obs_mean'].shape[0]
        action_dim = self.stats['act_mean'].shape[0]

        # Initialize training state (Optimizer)
        rng = jax.random.PRNGKey(0)
        self.wm_state = create_train_state(rng, learning_rate=1e-5, sensor_dim=sensor_dim, action_dim=action_dim) # Low LR for stability
        self.wm_state = self.wm_state.replace(params=params)
        
        # JIT compile the gradient descent logic
        self.fast_update = jax.jit(train_step)
        
        # History for domain detection
        self.errors = []
        self.p_values = []

        # total_size=1000: Takes 20 seconds to fill the queue
        # window_size=250: Detect changes based on the last 5 seconds of data
        self.detector = KSDriftDetector(total_size=1000, window_size=250, p_threshold=1e-3)

    def control_loop(self, obs, action, next_obs):
        # Predict what should have happened

        norm_obs = (obs - self.stats['obs_mean']) / self.stats['obs_std']
        norm_act = (action - self.stats['act_mean']) / self.stats['act_std']

        norm_delta = self.wm_state.apply_fn(
            {'params': self.wm_state.params}, norm_obs, norm_act
        )

        # Denormalize to check error in real units (e.g. degrees per second)
        pred_next_obs = obs + (norm_delta * self.stats['delta_std']) + self.stats['delta_mean']
        
        # Calculate error
        error = jp.mean((pred_next_obs - next_obs) ** 2)
        self.errors.append(error)
        
        # Update Detector
        is_drift, p_val = self.detector.update(error)
        
        self.p_values.append(p_val)
        if is_drift:
            print(f"!!! DOMAIN CHANGE DETECTED !!! p={p_val:.2e}.")

        if len(self.errors) % 500 == 0:
            #plt.hist(self.errors, bins=25, range=(0, 2))
            #plt.title("Error distribution on flat ground (Zoomed 0-2)")
            plt.plot(-np.log10(self.p_values))
            plt.title("log10(P-value) history")
            plt.show()

def interactive_visualization(env, jit_inference, controller=RobotController(), resetNum=-1):
    """
    Opens an interactive MuJoCo viewer for a JAX-based environment.
    
    Args:
        env: The mujoco_playground environment (unwrapped or wrapped).
        params: The trained policy parameters.
        inference_fn: The function make_inference_fn(params, deterministic=True).
    """

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

    jit_reset = env.jit_reset
    jit_step = env.jit_step

    # Initialize the Simulation State
    rng, key1 = jax.random.split(rng)
    state = jit_reset(rng)
    
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
                state = jit_reset(rng)
                reset_timer = 0
                resetNum -= 1

                if resetNum == 0:
                    break

            # Instruct robot to go always forwards
            state.info["command"] = jp.array([1., 0., 0.])

            step_start = time.time()
            rng, act_rng = jax.random.split(rng)
            
            # Run Policy
            action = jit_inference(state.obs, act_rng)[0]

            # Step environment
            prev_obs = state.obs["wm_state"]
            state = jit_step(state, action)
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
    impl = "jax"

    env = load_env(env_name, impl)
    rough_env = load_env("Go2StrollRoughTerrain", impl)

    obs_shape, act_shape = env.observation_size, env.action_size

    # Load network topology
    ppo_params = locomotion_params.brax_ppo_config(env_name, impl)

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
    jit_inference = jax.jit(inference_fn)

    controller = RobotController()

    interactive_visualization(env, jit_inference, controller, resetNum=2)
    interactive_visualization(rough_env, jit_inference, controller, resetNum=2)
    interactive_visualization(env, jit_inference, controller, resetNum=2)
    interactive_visualization(rough_env, jit_inference, controller, resetNum=2)

if __name__ == "__main__":
    main()