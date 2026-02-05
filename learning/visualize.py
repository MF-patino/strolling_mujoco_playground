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

# Work in progress controller for the robot
# TODO: integrate the policy loading logic into the controller
class RobotController:
    def __init__(self):
        # 1. Load Pre-trained Weights
        with open(MODEL_SAVE_PATH, 'rb') as f:
            params = pickle.load(f)
        with open(STATS_SAVE_PATH, 'rb') as f:
            self.stats = pickle.load(f)
            
        sensor_dim = self.stats['obs_mean'].shape[0]
        action_dim = self.stats['act_mean'].shape[0]

        # Initialize Training State (Optimizer)
        rng = jax.random.PRNGKey(0)
        self.wm_state = create_train_state(rng, learning_rate=1e-5, sensor_dim=sensor_dim, action_dim=action_dim) # Low LR for stability
        self.wm_state = self.wm_state.replace(params=params)
        
        # JIT compile the gradient descent logic
        self.fast_update = jax.jit(train_step)
        
        # History for domain detection
        self.errors = []

    def control_loop(self, obs, action, next_obs):
        # Predict what should have happened

        norm_obs = (obs - self.stats['obs_mean']) / self.stats['obs_std']
        norm_act = (action - self.stats['act_mean']) / self.stats['act_std']

        # Predict
        norm_delta = self.wm_state.apply_fn(
            {'params': self.wm_state.params}, norm_obs, norm_act
        )

        # Denormalize to check error in REAL units (e.g. degrees per second)
        pred_next_obs = obs + (norm_delta * self.stats['delta_std']) + self.stats['delta_mean']
        
        # Calculate Error
        error = jp.mean((pred_next_obs - next_obs) ** 2)
        self.errors.append(error)
        
        if len(self.errors) == 500:
            plt.hist(self.errors, bins=25, range=(0, 2))
            plt.title("Error Distribution on Flat Ground (Zoomed 0-2)")
            plt.show()

def interactive_visualization(env, inference_fn):
    """
    Opens an interactive MuJoCo viewer for a JAX-based environment.
    
    Args:
        env: The mujoco_playground environment (unwrapped or wrapped).
        params: The trained policy parameters.
        inference_fn: The function make_inference_fn(params, deterministic=True).
    """
    controller = RobotController()

    # Get the underlying standard MuJoCo model for the viewer
    if hasattr(env, 'mj_model'):
        model = env.mj_model
    else:
        # Fallback if environment is wrapped, try to access via unwrapped
        model = env.unwrapped.mj_model
        
    data = mujoco.MjData(model)

    # 2. Prepare JAX functions (jit them for performance)
    # We use a specific RNG key for the viz
    rng = jax.random.PRNGKey(0)
    
    # We need a single instance, not vmapped, but playground envs might 
    # expect batched inputs if configured that way. 
    # Assuming standard eval_env here.
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference = jax.jit(inference_fn)

    # 3. Initialize the Simulation State
    rng, key1 = jax.random.split(rng)
    state = jit_reset(rng)
    
    control_dt = getattr(env, 'dt', model.opt.timestep)
    
    timer = 0
    print(f"Simulation running at Control DT: {control_dt:.4f}s")

    # 4. Launch the Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Initialize viewer camera if needed
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.5]
        
        step_start = time.time()
        
        while viewer.is_running():
            if timer >= 10:
                rng, key1 = jax.random.split(rng)
                state = jit_reset(rng)
                timer = 0
                #print("Resetting")

            state.info["command"] = jp.array([1., 0., 0.])
            step_start = time.time()

            # --- A. INFERENCE & PHYSICS (GPU/JAX) ---
            rng, act_rng = jax.random.split(rng)
            
            # Run Policy
            action = jit_inference(state.obs, act_rng)[0]

            # Step Environment
            # Note: We don't use 'do_rollout' here because we want to loop indefinitely
            prev_obs = state.obs["wm_state"]
            state = jit_step(state, action)
            controller.control_loop(prev_obs, action, state.obs["wm_state"])

            # Ensure computation is done before we try to read it
            state.data.qpos.block_until_ready()
            # --- B. SYNC TO CPU VIEWER ---
            
            # Extract qpos/qvel from JAX to Numpy (CPU)
            # Copy to the viewer's data structure
            # We use np.array() to enforce the transfer from JAX device array to CPU numpy
            data.qpos[:] = np.array(state.data.qpos)
            data.qvel[:] = np.array(state.data.qvel)

            # Forward kinematics (compute world positions of geoms based on qpos)
            mujoco.mj_forward(model, data)

            # Sync the viewer
            viewer.sync()

            timer += control_dt
            elapsed = time.time() - step_start
            # Sleep only the remaining time to match real-time
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)


from brax.training.agents.ppo import networks as ppo_networks
import functools
from mujoco_playground import registry
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
from mujoco_playground.config import locomotion_params

def main():
    '''
    Loads the omni-directional policy network and environment for the Go2 Stroll environment.
    Then, it simulates it in the Mujoco interactive viewer using JAX.
    '''

    env_name = "Go2StrollFlatTerrain"
    checkpoint_path = "/home/marcos/Escritorio/mujoco_playground/logs/Go2StrollFlatTerrain-20260122-183735/checkpoints/000200540160"
    impl = "jax"

    env_cfg = registry.get_default_config(env_name)
    env_cfg["impl"] = impl

    ppo_params = locomotion_params.brax_ppo_config(env_name, impl)
    ppo_params.num_timesteps = 0

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
    
    env = registry.load(env_name, config=env_cfg)
    ppo_network = network_factory(
      env.observation_size, env.action_size, preprocess_observations_fn=normalize
    )

    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

    print(f"Loading weights from: {checkpoint_path}")
    params = checkpoint.load(checkpoint_path)

    inference_fn = make_inference_fn(params, deterministic=True)
    interactive_visualization(env, inference_fn)

if __name__ == "__main__":
    main()