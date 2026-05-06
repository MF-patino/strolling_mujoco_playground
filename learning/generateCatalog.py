import os
import jax.numpy as jp
from controller.offline_controller import OfflineRobotController
from worldModel.common import MODELS_ROOT
from visualize_adaptation import load_env, IMPL

ALL_FROM_SCRATCH = False

def main():
    '''
    Generates a catalog of policies trained from scratch for all environments,
    as well as the adapted policies using the flat terrain policy trained from scratch as a checkpoint.
    '''

    flat_env = load_env("Go2StrollFlatTerrain", IMPL)
    rough_env = load_env("Go2StrollRoughTerrain", IMPL)
    slippery_env = load_env("Go2StrollSlipperyTerrain", IMPL)
    env_blocked = load_env("Go2StrollFlatTerrain", IMPL, breakLeg=True)

    obs_shape, act_shape = flat_env.observation_size, flat_env.action_size

    # Create a new catalog
    os.makedirs(MODELS_ROOT, exist_ok=True)
    
    all_envs = [flat_env, rough_env, slippery_env, env_blocked]
    pol_base_names = ["FlatTerrain", "RoughTerrain", "SlipperyTerrain", "BlockedKnee"]
    # Training from scratch
    for base_name, env in zip(pol_base_names, all_envs):
        controller = OfflineRobotController(obs_shape, act_shape, initial_pair=None, 
                                            generatePlots = False, cmd = jp.array([1., 0., 0.]))
        controller.env = env
        # When base_policy_name = None, we train from scratch
        controller.adapt_policy(base_policy_name=None, key_name=base_name)

        if not ALL_FROM_SCRATCH:
            # Only train from scratch FlatTerrain
            break

    if ALL_FROM_SCRATCH:
        return

    # Adapting from FlatTerrain policy
    for base_name, env in zip(pol_base_names[1:], all_envs[1:]):
        controller = OfflineRobotController(obs_shape, act_shape, initial_pair=None, 
                                            generatePlots = False, cmd = jp.array([1., 0., 0.]))
        controller.env = env
        controller.adapt_policy(pol_base_names[0], key_name=base_name)
    
    # Adapting all combinations
    def getEnv(pol_name):
        if "Rough" in pol_name:
            return rough_env
        elif "Slippery" in pol_name:
            return slippery_env
        elif "Blocked" in pol_name:
            return env_blocked
        elif "Flat" in pol_name:
            return flat_env
        else:
            raise Exception("Could not find environment.")

    initialCatalog = os.listdir(MODELS_ROOT)
    for name in initialCatalog:
        env_name = name.split("_AdaptedFrom_")[0]
        env = getEnv(env_name)

        for base_name in initialCatalog:
            if name == base_name or "Adapted" not in base_name:
                continue
            
            controller = OfflineRobotController(obs_shape, act_shape, initial_pair=None, 
                                            generatePlots = False, cmd = jp.array([1., 0., 0.]))
            controller.env = env
            controller.adapt_policy(base_name, key_name=env_name)

if __name__ == "__main__":
    main()