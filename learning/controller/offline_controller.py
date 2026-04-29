from controller.robot_controller import RobotController
import os

import jax
import jax.numpy as jp
import numpy as np

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params
from mujoco_playground import registry
import functools
from mujoco_playground import wrapper

from brax.training.agents.ppo import checkpoint
from brax.training.acme import running_statistics

from worldModel.common import WM_DS_PATH, POL_PATH, MODELS_ROOT
from worldModel.rollout_saver import WorldModelRolloutSaver
from worldModel.train_world_model import trainWM, load_dataset

from controller.plots import PLOT_DATA_DIR, TRAIN_DATA_SUBDIR
IMPL = "jax"

class OfflineRobotController(RobotController):
    def __init__(self, obs_shape, act_shape, initial_pair=None, jit_inference=None,
                 generatePlots = True, cmd = jp.array([1., 0., 0.])):
        RobotController.__init__(self, obs_shape, act_shape, initial_pair, jit_inference, generatePlots, cmd)

    def adapt_policy(self, base_policy_name):
        basePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        
        print("--- STARTING HEADLESS ADAPTATION FROM " + (base_policy_name if base_policy_name else "SCRATCH") + " ---")

        # Setup paths and parameters
        base_ckpt_path = basePath + "/" + POL_PATH.format(env_name=base_policy_name) if base_policy_name else None
        new_policy_name = str(len(os.listdir(MODELS_ROOT)))
        if base_policy_name:
            new_policy_name += "_AdaptedFrom_" + base_policy_name

        new_ckpt_path = basePath + "/" + POL_PATH.format(env_name=new_policy_name)

        # Setup logs folder
        plot_dir = os.path.join(basePath, PLOT_DATA_DIR, TRAIN_DATA_SUBDIR)
        os.makedirs(plot_dir, exist_ok=True)
        plot_file_path = os.path.join(plot_dir, f"{new_policy_name}.npz")

        self.ppo_params.num_timesteps = 85_000_000
        self.ppo_params.num_evals = int(self.ppo_params.num_timesteps/1_000_000)
        
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **self.ppo_params.network_factory)

        # Policy parameter checkpointer
        captured_params = [None]
        def policy_params_fn(current_step, make_policy, params):
            # This continuously grabs the latest network weights during training
            captured_params[0] = params

        training_history = {
            'steps': [],
            'reward_mean': [],
            'reward_std':[]
        }
        # Convergence Checker (progress_fn)
        best_reward = 0
        evals_since_best = 0

        def progress_fn(num_steps, metrics):
            nonlocal best_reward, evals_since_best

            reward_mean = metrics.get('eval/episode_reward', 0.0)
            reward_std = metrics.get('eval/episode_reward_std', 0.0)
            
            training_history['steps'].append(num_steps)
            training_history['reward_mean'].append(reward_mean)
            training_history['reward_std'].append(reward_std)

            print(f"Step {num_steps}: Reward = {reward_mean:.2f} ± {reward_std:.2f}")
            
            # Early stopping logic for checking convergence
            if reward_mean > best_reward:
                best_reward = reward_mean
                evals_since_best = 0
            else:
                evals_since_best += 1

            if evals_since_best > 10:
                pass
                #print(f"Convergence reached at step {num_steps}!")
                #raise StopIteration("Converged") # Hack to cleanly break Brax's jax.lax.scan loop

        # Train the Policy offline (This will block the viewer)
        final_params = None
        try:
            make_inference_fn, final_params, _ = ppo.train(
                environment=self.env,
                network_factory=network_factory,
                restore_checkpoint_path=base_ckpt_path,
                policy_params_fn=policy_params_fn,
                progress_fn=progress_fn,
                seed=0,
                wrap_env_fn=wrapper.wrap_for_brax_training,
                **{k: v for k, v in self.ppo_params.items() if k not in ['network_factory']}
            )
        except StopIteration:
            # We aborted training manually, so grab the params we caught in the background
            final_params = captured_params[0]
            
            # Because ppo.train aborted, it didn't return make_inference_fn. We must rebuild it.
            normalize = running_statistics.normalize if self.ppo_params.normalize_observations else lambda x, y: x
            ppo_network = network_factory(
                self.env.observation_size, self.env.action_size, preprocess_observations_fn=normalize
            )
            make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        
        print(f"Saving training history for plotting to: {plot_file_path}")
        np.savez_compressed(
            plot_file_path,
            steps=np.array(training_history['steps']),
            reward_mean=np.array(training_history['reward_mean']),
            reward_std=np.array(training_history['reward_std'])
        )
        print(f"Saving adapted policy to: {new_ckpt_path}")
        # Ensure the target directory exists before saving
        os.makedirs(new_ckpt_path, exist_ok=True)

        ckpt_config = checkpoint.network_config(
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            normalize_observations=self.ppo_params.normalize_observations,
            network_factory=network_factory,
        )
        checkpoint.save(new_ckpt_path, self.ppo_params.num_timesteps, final_params, ckpt_config)
        
        # Move policy files up into the parent directory
        dir = os.listdir(new_ckpt_path)[0]
        for filename in os.listdir(os.path.join(new_ckpt_path, dir)):
            os.rename(os.path.join(new_ckpt_path, dir, filename), os.path.join(new_ckpt_path, filename))   
        os.rmdir(os.path.join(new_ckpt_path, dir)) 
        
        # Load the newly trained policy into active memory
        inference_fn = make_inference_fn(final_params, deterministic=True)

        # Configuration for world model dataset collection
        wm_saver = WorldModelRolloutSaver(
            env=self.env,
            episode_length=self.ppo_params.episode_length,
            num_envs=128,
            data_dir=WM_DS_PATH.format(env_name=new_policy_name),
            deterministic=False,
        )

        # Synthesis of WM dataset
        wm_saver.set_make_policy(make_inference_fn)
        wm_saver.dump_rollout(final_params, 0)
        
        # Training a WM from the generated WM dataset
        wm_info = trainWM(new_policy_name)

        self.integrateNewModelPair(new_policy_name, (wm_info, inference_fn))

        # Set the model pair as the active pair so the viewer immediately uses it
        self.inference = self.policies[new_policy_name]
        self.active_wm = self.wm_dict[new_policy_name]

    def integrateNewModelPair(self, pair_name, model_pair):
        wm_info, inference_fn = model_pair

        # Create a new WM error category
        self.errors[pair_name] = []

        # Include models in internal data structures
        self.wms.append(wm_info)
        self.wm_dict[pair_name] = wm_info

        self.policies[pair_name] = jax.jit(inference_fn)

        # On recomputing the policy embeddings:
        # We can make this efficient by only computing those entries in the inaffinity
        # matrix that are absent (one extra column to the right, an extra row downwards)
        _, wm_state, wm_stats = wm_info
        new_column = []
        for env_name in self.pol_names:
            datasetPath = WM_DS_PATH.format(env_name=env_name)
            obs_data, act_data, next_data = load_dataset(datasetPath)
            
            # Calculate mean error of the new WM on the cataloged policy's data
            err = jp.mean(jp.linalg.norm(self.getPredictionWM(wm_state, wm_stats, obs_data, act_data) - next_data, axis=1))
            
            # Calculate surprise by subtracting the OLD policy's native baseline noise
            new_column.append(jp.abs(err - self.native_errors[env_name][0]))
            
        # Reshape to column vector (N, 1) and append to the existing matrix (N, N) -> (N, N+1)
        new_column = jp.array(new_column).reshape(-1, 1)

        self.pol_names.append(pair_name)
        if len(self.pol_names) == 1:
            self.inaffinity_matrix = jp.stack([self.computePolicyEmbedding(name) for name in self.pol_names])
        else:
            updated_embeddings = jp.hstack([self.inaffinity_matrix, new_column])

            # Compute the complete embedding for the new policy
            new_embedding = self.computePolicyEmbedding(pair_name)

            self.inaffinity_matrix = jp.vstack([updated_embeddings, new_embedding])

        print(f"Adjusted raw inaffinity matrix:\n{self.inaffinity_matrix}")

        # Re-normalize the new inaffinity matrix to project all points onto the hypersphere
        if len(self.pol_names) > 1:
            self.normalizePolicyEmbeddings()
        