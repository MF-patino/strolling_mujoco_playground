from brax.training.agents.ppo import networks as ppo_networks
import functools

import numpy as np
import jax
import jax.numpy as jp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
from mujoco_playground.config import locomotion_params

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

from worldModel.train_world_model import create_train_state, train_step, load_dataset, ALL_ENVS
import os
import pickle
from collections import deque
from controller.ks_detector import KSDriftDetector

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
IMPL = "jax"

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

        self.inaffinity_matrix = jp.stack([self.computePolicyEmbedding(name) for name in ALL_ENVS])

        print(f"Raw inaffinity matrix:\n{self.inaffinity_matrix}")

        norm_mat = self.inaffinity_matrix / jp.linalg.norm(self.inaffinity_matrix, axis=1, keepdims=True) 

        # Normalization
        self.policy_embeddings = {
            name: embedding for name, embedding in zip(ALL_ENVS, norm_mat)
        }

        print(f"Normalized policy embeddings:")
        for pol_name in self.policy_embeddings:
            print(f"{pol_name}: {self.policy_embeddings[pol_name]}")

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
        
        self.sampling = False

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
        basePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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

    def getErrorWM(self, name, wm, stats, obs, action, next_obs):
        pred_next_obs = self.getPredictionWM(wm, stats, obs, action)
        
        # Calculate error
        error = jp.mean(jp.square(pred_next_obs - next_obs))
        alpha = 0.1
        smooth_error = error if len(self.errors[name]) == 0 else alpha * error + (1-alpha) * self.errors[name][-1]
        
        self.errors[name].append(smooth_error)

        return error

    def computePolicyEmbedding(self, env_name):
        '''
        The embeddings of a policy are its affinity coefficients to each environment.
        '''

        root = f"world_models/{env_name}/"
        datasetPath = root + "world_model_dataset"

        # Load data
        obs_data, act_data, next_data = load_dataset(datasetPath)

        mean_errors = jp.array([
            jp.sqrt(jp.mean(jp.square(self.getPredictionWM(wm, stats, obs_data, act_data) - next_data)))
            for _, wm, stats in self.wms
        ])

        env_index = ALL_ENVS.index(env_name)
        # Compute the extra "surprise" by subtracting the baseline noise
        embedding = jp.abs(mean_errors - mean_errors[env_index])
        
        return embedding
        
    def predictPolicyScore(self, name, active_name):
        # GP predicts the expected reward (negative error) and uncertainty
        pol_emb = self.policy_embeddings[name]
        active_emb = self.policy_embeddings[active_name]
        mean, std = self.gp.predict([pol_emb], return_std=True)
                        
        # UCB formula: Exploit (mean) + Explore (Kappa * std) - Safety constraint (Gamma * policy distance)

        # It is dangerous to rollout policies that are too different together for stability reasons.
        # With the safety constraint, the cost of transitioning from one to the next is lower.
        # If not, the robot may switch directly from flat to slippery on rough terrain, falling over.
        cos_dist = 1 - jp.dot(pol_emb, active_emb) / (jp.linalg.norm(pol_emb) * jp.linalg.norm(active_emb))
        kappa = 1.5; gamma = .8#2.
        ucb_score = -mean[0] + kappa * std[0] - gamma * cos_dist#* jp.exp(-gamma * cos_dist)
        return ucb_score, mean, std
    
    def getNextPolicy(self, active_name):
        # Fit the GP with the new real-world data
        noise = 1e-06

        self.gp = GaussianProcessRegressor(kernel=RBF(), normalize_y=True, alpha=self.alpha+noise)
        self.gp.fit(self.X_train, self.y_train)

        best_ucb = -float('inf')
        next_name = None
        
        for name in self.policies:
            ucb_score, mean, std = self.predictPolicyScore(name, active_name)
            print(f"{name} Mean: {mean[0]}±{std[0]}. Score {ucb_score}")
            
            if ucb_score > best_ucb:
                best_ucb = ucb_score
                next_name = name

        return next_name

    def control_loop(self, obs, action, next_obs):
        if not self.deploy:
            return
        
        obs, next_obs = obs["wm_state"], next_obs["wm_state"]
        
        self.buffer.append((obs, action, next_obs))

        if self.sampling:
            prev_active_name = self.active_wm[0]
            self.active_wm = None
            for name in self.samples2collect:
                sample_num = len(self.sampled_errors[name])
                if sample_num < self.samples2collect[name]:
                    if prev_active_name != name:
                        print(f"Sampling {name}.")

                    self.active_wm = self.wm_dict[name]
                    self.inference = self.policies[name]
                    break

            if self.active_wm is None:
                next_name = self.getNextPolicy(prev_active_name)

                if self.iteration < self.max_iterations:
                    # Forget data that will be stale in the next iteration
                    # As the policy chosen by the GP will be resampled, the mean error and std will change
                    mask = (self.X_train != self.policy_embeddings[next_name]).any(axis=1)
                    self.X_train, self.y_train, self.alpha = self.X_train[mask], self.y_train[mask], self.alpha[mask]

                    # Update rule: collect more samples of a policy that improves WM error (to stabilize the robot)
                    # Also, don't collect more samples of policies that increase WM error (as it destabilizes the robot)
                    
                    # We collect more samples of a promising policy to give us more evidence in favour or to the contrary
                    # that the policy is actually better than the rest
                    self.samples2collect[next_name] = self.samples2collect.get(next_name, 0) + self.increment_samples

                    self.active_wm = self.wm_dict[next_name]
                    self.inference = self.policies[next_name]
                    self.iteration += 1

                    print(f"Beginning iteration {self.iteration} with {next_name}.")
                else:
                    print(f"Converged on {next_name}.")
                    self.active_wm = self.wm_dict[next_name]
                    self.inference = self.policies[next_name]
                    self.sampling = False

        active_name, wm, stats = self.active_wm
        error = self.getErrorWM(active_name, wm, stats, obs, action, next_obs)
        #[self.getErrorWM(name, wm, stats, obs, action, next_obs) for name, wm, stats in self.wms if name != active_name]

        if self.sampling:
            self.sampled_errors[active_name].append(error)

            # When sampling batch is over for the policy, compute its new GP datapoint
            if len(self.sampled_errors[active_name]) == self.samples2collect[active_name]:
                samples = jp.array(self.sampled_errors[active_name])

                self.X_train = np.vstack([self.X_train, self.policy_embeddings[active_name]])
                self.y_train = np.append(self.y_train, jp.mean(samples))
                self.alpha = np.append(self.alpha, jp.var(samples)/len(samples))

            return
        
        # Update detector
        is_drift, statistic = self.detector.update(error)

        step = len(self.detector.stat_values)
        
        if is_drift:
            idx = step - 1
            self.drift_indices.append(idx)
            print(f"!!! DOMAIN CHANGE DETECTED !!! statistic={statistic:.2e}.")
            self.sampled_errors = {name: [] for name, _, _ in self.wms}

            # Policies are safely sampled 15 times (.3 seconds) initially. 
            # 20 (.4 seconds) is too much for slippery in rough terrain: robot falls.
            self.increment_samples = 15

            # Assign last obtained errors so that we may be able to compare other policies
            # against the originally active one without having to roll it out
            samples = self.errors[active_name][-self.increment_samples*3:]
            self.sampled_errors[active_name] = samples

            self.X_train = np.array([self.policy_embeddings[active_name]])
            samples = jp.array(samples)
            self.y_train = np.array([jp.mean(samples)])
            self.alpha = np.array([jp.var(samples)/len(samples)])

            self.iteration = 1
            # Should converge after 5 iterations
            self.max_iterations = 5

            next_name = self.getNextPolicy(active_name)

            # When checking out different policies, the amount of samples to collect of each
            self.samples2collect = {next_name: self.increment_samples}
            self.samples2collect[active_name] = len(samples)

            self.sampling = True

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