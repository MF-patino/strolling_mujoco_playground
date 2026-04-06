from mujoco_playground import registry

import time
import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer
from controller.offline_controller import RobotController, OfflineRobotController

IMPL = "jax"
    
def load_env(env_name, impl, breakLeg=False):
    env_cfg = registry.get_default_config(env_name)
    env_cfg["impl"] = impl
    if breakLeg:
        env_cfg["broken_leg"] = True

    env = registry.load(env_name, config=env_cfg)

    env.jit_reset = jax.jit(env.reset)
    env.jit_step = jax.jit(env.step)
    env.name = env_name

    return env

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

    controller.setEnv(env)

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
            controller.control_loop(prev_obs, action, state)

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

    flat_env = load_env(env_name, IMPL)
    rough_env = load_env("Go2StrollRoughTerrain", IMPL)
    slippery_env = load_env("Go2StrollSlipperyTerrain", IMPL)
    env_broken = load_env(env_name, IMPL, breakLeg=True)

    obs_shape, act_shape = flat_env.observation_size, flat_env.action_size

    cmds = [jp.array([1., 0., 0.]), jp.array([.6, 0., 0.]), jp.array([.25, 0., 0.])]
    for cmd in cmds:
        controller = RobotController(obs_shape, act_shape, initial_env=env_name, 
                                            generatePlots = False, cmd = cmd)

        interactive_visualization(flat_env, controller=controller, resetNum=1)
        interactive_visualization(env_broken, controller=controller, resetNum=1)
        interactive_visualization(flat_env, controller=controller, resetNum=1)
        interactive_visualization(slippery_env, controller=controller, resetNum=1)
        interactive_visualization(flat_env, controller=controller, resetNum=1)
        interactive_visualization(rough_env, controller=controller, resetNum=1)
        interactive_visualization(flat_env, controller=controller, resetNum=1)
        interactive_visualization(rough_env, controller=controller, resetNum=1)
        interactive_visualization(flat_env, controller=controller, resetNum=1)
        interactive_visualization(slippery_env, controller=controller, resetNum=1)
        interactive_visualization(rough_env, controller=controller, resetNum=1)
        interactive_visualization(flat_env, controller=controller, resetNum=1)
        interactive_visualization(rough_env, controller=controller, resetNum=2)
        interactive_visualization(slippery_env, controller=controller, resetNum=1)
        controller.export_history(f"{cmd}.pkl")

if __name__ == "__main__":
    main()