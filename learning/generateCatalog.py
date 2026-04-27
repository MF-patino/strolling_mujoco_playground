import os, shutil
import jax.numpy as jp
from controller.offline_controller import OfflineRobotController
from worldModel.common import MODELS_ROOT
from visualize_adaptation import load_env, IMPL
from pathlib import Path

ALL_FROM_SCRATCH = False

def main():
    '''
    Generates a catalog of policies trained from scratch for all environments,
    as well as the adapted policies using the flat terrain policy trained from scratch as a checkpoint.
    '''

    env_name = "Go2StrollFlatTerrain"

    flat_env = load_env(env_name, IMPL)
    rough_env = load_env("Go2StrollRoughTerrain", IMPL)
    slippery_env = load_env("Go2StrollSlipperyTerrain", IMPL)
    env_blocked = load_env(env_name, IMPL, breakLeg=True)

    obs_shape, act_shape = flat_env.observation_size, flat_env.action_size

    # Wiping clean the current catalog
    if os.path.exists(MODELS_ROOT):
        shutil.rmtree(MODELS_ROOT)
    os.makedirs(MODELS_ROOT)
    
    all_envs = [flat_env, rough_env, slippery_env, env_blocked]
    pol_base_names = ["FlatTerrain", "RoughTerrain", "SlipperyTerrain", "BlockedKnee"]
    # Training from scratch
    for base_name, env in zip(pol_base_names, all_envs):
        controller = OfflineRobotController(obs_shape, act_shape, initial_pair=None, 
                                            generatePlots = False, cmd = jp.array([1., 0., 0.]))
        controller.setEnv(env)
        controller.pol_names = []
        last_pol = sorted(Path(MODELS_ROOT).iterdir(), key=os.path.getctime)[-1]
        shutil.move(last_pol, last_pol.with_stem(base_name))

        if not ALL_FROM_SCRATCH:
            # Only train from scratch FlatTerrain
            break

    # Adapting from FlatTerrain policy
    for base_name, env in zip(pol_base_names[1:], all_envs[1:]):
        controller = OfflineRobotController(obs_shape, act_shape, initial_pair=None, 
                                            generatePlots = False, cmd = jp.array([1., 0., 0.]))
        controller.setEnv(env)
        controller.adapt_policy(pol_base_names[0])
        last_pol = sorted(Path(MODELS_ROOT).iterdir(), key=os.path.getctime)[-1]
        shutil.move(last_pol, last_pol.with_stem(f"{base_name}_AdaptedFrom_{pol_base_names[0]}"))
    
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
        env = getEnv(name)
        if name == "FlatTerrain":
            continue
        for base_name in initialCatalog:
            if name == base_name or "Adapted" not in base_name:
                continue
            
            controller = OfflineRobotController(obs_shape, act_shape, initial_pair=None, 
                                            generatePlots = False, cmd = jp.array([1., 0., 0.]))
            controller.setEnv(env)
            controller.adapt_policy(base_name)
            last_pol = sorted(Path(MODELS_ROOT).iterdir(), key=os.path.getctime)[-1]
            shutil.move(last_pol, last_pol.with_stem(f"{name}_AdaptedFrom_{base_name}"))

if __name__ == "__main__":
    main()