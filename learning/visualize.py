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
import os

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
        #self.buffer.extend(data[-self.window_size:])
        self.adwin._reset()
    
def load_env(env_name, impl):
    env_cfg = registry.get_default_config(env_name)
    env_cfg["impl"] = impl

    env = registry.load(env_name, config=env_cfg)

    env.jit_reset = jax.jit(env.reset)
    env.jit_step = jax.jit(env.step)

    return env

# Work in progress controller for the robot
class RobotController:
    def __init__(self, obs_shape, act_shape, initial_env=None, jit_inference=None):
        self.cmd = jp.array([1., 0., 0.])
        self.wms = []
        self.policies = {}

        # If a single inference (policy) is given, then we don't deploy the full
        # online adaptation system
        self.deploy = jit_inference is None

        if jit_inference is not None:
            self.inference = jit_inference
        
        if not self.deploy:
            return
        
        self.loadPolicies(initial_env, obs_shape, act_shape)
        self.loadWorldModels(initial_env)
        print(f"Starting up with {initial_env} world model & policy.")

        # JIT compile the gradient descent logic
        self.fast_update = jax.jit(train_step)
        
        # History of errors for plotting 
        self.errors = {name: [] for name, _, _ in self.wms}
        self.drift_indices = []

        # total_size=1000: Takes 20 seconds to fill the queue
        # window_size=250: Detect changes based on the last 5 seconds of data
        window_size = 100
        self.detector = KSDriftDetector(total_size=1000, window_size=window_size, adwin_delta=5e-2)
        self.buffer = deque(maxlen=window_size)
        
        self.to_sample = None

    def loadWorldModels(self, initial_env):
        self.wm_dict = {}

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
            self.wm_dict[env_name] = (env_name, wm_state.replace(params=params), stats)

            # The active world model is the one corresponding to the environment we launch the robot in
            if env_name == initial_env:
                self.active_wm = self.wms[-1]

    def loadPolicies(self, initial_env, obs_shape, act_shape):
        basePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        for env_name in ALL_ENVS:
            checkpoint_path = basePath + f"/world_models/{env_name}/policy"

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
            self.policies[env_name] = jax.jit(inference_fn)

        # The active policy is the one corresponding to the environment we launch the robot in
        self.inference = self.policies[initial_env]
        
    def getPredictionWM(self, wm, stats, obs, action):
        # Predict what should have happened
        norm_obs = (obs - stats['obs_mean']) / stats['obs_std']
        norm_act = (action - stats['act_mean']) / stats['act_std']

        norm_delta = wm.apply_fn(
            {'params': wm.params}, norm_obs, norm_act
        )

        # Denormalize to check error in real units (e.g. degrees per second)
        return obs + (norm_delta * stats['delta_std']) + stats['delta_mean']
    
    def getMetricWM(self, wm, stats):
        batch_obs, batch_act, batch_next = zip(*self.buffer)
        
        # Stack into JAX arrays (Shape: [Batch_Size, Dim])
        obs_arr = jp.stack(batch_obs)
        act_arr = jp.stack(batch_act)
        next_arr = jp.stack(batch_next)
        
        return jp.mean((self.getPredictionWM(wm, stats, obs_arr, act_arr) - next_arr) ** 2)

    def getErrorWM(self, name, wm, stats, obs, action, next_obs):
        pred_next_obs = self.getPredictionWM(wm, stats, obs, action)
        
        # Calculate error
        error = jp.mean((pred_next_obs - next_obs) ** 2)
        alpha = 0.1
        smooth_error = error if len(self.errors[name]) == 0 else alpha * error + (1-alpha) * self.errors[name][-1]
        
        self.errors[name].append(smooth_error)

        return error

    def control_loop(self, obs, action, next_obs):
        if not self.deploy:
            return
        
        obs, next_obs = obs["wm_state"], next_obs["wm_state"]
        
        self.buffer.append((obs, action, next_obs))

        # TODO: implement Multi-Armed Bandit with policy embeddings
        if self.to_sample is not None:
            prev_name = self.active_wm[0]
            self.active_wm = None
            for try_wm in self.to_sample:
                name = try_wm[0]
                sample_num = len(self.sampled_errors[name])
                if sample_num < self.samples2collect[name]:
                    if prev_name != name:
                        print(f"Sampling {name}.")
                    self.active_wm = try_wm
                    self.inference = self.policies[name]
                    break

            if self.active_wm is None:
                # Determine the policy that garnered more success (lowest mean WM error)
                errs = [(jp.mean(jp.array(self.sampled_errors[name])), name) for name in self.sampled_errors]
                best_err, best_name = min(errs)

                if self.iteration < self.max_iterations:
                    # Update rule: collect more samples of a policy that improves WM error (to stabilize the robot)
                    # Also, don't collect more samples of policies that increase WM error (as it destabilizes the robot)
                    
                    # We collect more samples of a promising policy to give us more evidence in favour or to the contrary
                    # that the policy is actually better than the rest
                    delta = lambda err: 15 if err == best_err else 0
                    self.samples2collect = {name: int(self.samples2collect[name] + delta(err)) for err, name in errs}
                    
                    self.active_wm = [(name, wm, stats) for name, wm, stats in self.to_sample if len(self.sampled_errors[name]) < self.samples2collect[name]][0]
                    next_name = self.active_wm[0]
                    self.inference = self.policies[next_name]
                    self.iteration += 1

                    print(f"Beginning iteration {self.iteration} with {next_name}.")
                else:
                    print(f"Converged on {best_name}.")
                    self.active_wm = self.wm_dict[best_name]
                    self.inference = self.policies[best_name]
                    self.to_sample = None

        active_name, wm, stats = self.active_wm
        error = self.getErrorWM(active_name, wm, stats, obs, action, next_obs)
        #[self.getErrorWM(name, wm, stats, obs, action, next_obs) for name, wm, stats in self.wms if name != active_name]

        if self.to_sample is not None:
            self.sampled_errors[active_name].append(error)
            return
        
        # Update detector
        is_drift, statistic = self.detector.update(error)

        step = len(self.detector.stat_values)
        
        if is_drift:
            idx = step - 1
            self.drift_indices.append(idx)
            print(f"!!! DOMAIN CHANGE DETECTED !!! statistic={statistic:.2e}.")

            wm_info = [(self.getMetricWM(wm, stats), name, wm, stats) for name, wm, stats in self.wms if name != active_name]
            #print([(error, name) for error, name, _, _ in wm_info])

            # We sort by mean WM error (this implicitly sorts the policies in relation to their similarity
            # with the active policy)
            self.to_sample = [self.active_wm] + [(name, wm, stats) for _, name, wm, stats in sorted(wm_info)]
            self.sampled_errors = {name: [] for name, _, _ in self.to_sample}

            # All policies are safely sampled 15 times (.3 seconds) initially. 
            # 20 (.4 seconds) is too much for slippery in rough terrain: robot falls.
            self.initial_samples = 15

            # Assign last obtained errors so that we may be able to compare other policies
            # against the originally active one without having to roll it out
            self.sampled_errors[active_name] = self.errors[active_name][-self.initial_samples:]

            # When checking out different policies, the amount of samples to collect of each
            self.samples2collect = {name: self.initial_samples for name, _, _ in self.to_sample}
            self.iteration = 1
            # Should converge after 5 iterations
            self.max_iterations = 5

        if is_drift and step > self.detector.min_samples:
            '''for wm_name in self.errors:
                plt.plot(self.errors[wm_name], label=f"{wm_name} WM errors")

            plt.xlabel("Time step")
            plt.title("WM error history")
            plt.legend()
            plt.tight_layout()
            plt.show()'''

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
            state.info["command"] = controller.cmd

            step_start = time.time()
            rng, act_rng = jax.random.split(rng)
            
            # Run Policy
            action = controller.inference(state.obs, act_rng)[0]

            # Step environment
            prev_obs = state.obs
            state = env.jit_step(state, action)
            controller.control_loop(prev_obs, action, state.obs)

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
    #checkpoint_path = "/home/marcos/Escritorio/mujoco_playground/logs/Go2StrollFlatTerrain-20260122-183735/checkpoints/000200540160"
    #checkpoint_path = "/home/marcos/Escritorio/mujoco_playground/logs/Go2StrollRoughTerrain-20260208-001847/checkpoints/000200540160"

    env = load_env(env_name, IMPL)
    rough_env = load_env("Go2StrollRoughTerrain", IMPL)
    #stairs_env = load_env("Go2StrollStairs", IMPL)
    slippery_env = load_env("Go2StrollSlipperyTerrain", IMPL)

    # TODO:
    # 
    # Entrenar rugoso desde llano
    # Comprobar que funciona en el G1

    obs_shape, act_shape = env.observation_size, env.action_size

    controller = RobotController(obs_shape, act_shape, initial_env=env_name)

    interactive_visualization(env, controller=controller, resetNum=1)
    interactive_visualization(rough_env, controller=controller, resetNum=1)
    interactive_visualization(env, controller=controller, resetNum=1)
    interactive_visualization(slippery_env, controller=controller, resetNum=1)
    interactive_visualization(env, controller=controller, resetNum=1)

if __name__ == "__main__":
    main()