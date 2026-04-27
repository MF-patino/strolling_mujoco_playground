from brax.training.agents.ppo import networks as ppo_networks
import functools

import numpy as np
import jax
import jax.numpy as jp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
from mujoco_playground.config import locomotion_params

from worldModel.common import MODELS_ROOT, WM_PATH, WM_STATS_PATH, WM_DS_PATH, POL_PATH
from worldModel.train_world_model import create_train_state, train_step, load_dataset

import os
import pickle
from controller.ks_detector import KSDriftDetector
import controller.plots as plots

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.decomposition import TruncatedSVD
IMPL = "jax"

@functools.partial(jax.jit, static_argnames=['apply_fn'])
def _jit_predict_wm(apply_fn, params, stats, obs, action):
    # Predict what should have happened
    norm_obs = (obs - stats['obs_mean']) / stats['obs_std']
    norm_act = (action - stats['act_mean']) / stats['act_std']

    norm_delta = apply_fn({'params': params}, norm_obs, norm_act)

    # Denormalize to check error in real units
    return obs + (norm_delta * stats['delta_std']) + stats['delta_mean']

class RobotController:
    def __init__(self, obs_shape, act_shape, initial_pair=None, jit_inference=None,
                 generatePlots = True, cmd = jp.array([1., 0., 0.])):
        self.generatePlots = generatePlots
        self.cmd = cmd
        self.wms = []
        self.policies = {}

        # If a single inference (policy) is given, then we don't deploy the full
        # online adaptation system
        self.deploy = jit_inference is None

        if jit_inference is not None:
            self.inference = jit_inference

        if not self.deploy:
            return

        self.pol_names = os.listdir(MODELS_ROOT)

        # Histories for plotting
        self.smooth_errors = {name: [] for name in self.pol_names}
        self.env_changes = []
        self.drift_indices = []
        self.contact_history = []
        self.policy_history = []
        self.gp_states = []

        # KS-ADWIN detector configuration:
        # total_size=1000: 20 seconds of total memory
        # window_size=100: Detect changes based on the last 2 seconds of data
        self.detector = KSDriftDetector(total_size=1000, window_size=100, adwin_delta=.01)

        # GP search parameters:
        self.errors = {name: [] for name in self.pol_names}
        # Should converge after 5 iterations (empirically tested)
        self.max_iterations = 5
        self.sampling = False
        # Policy embeddings will have a maximum of 4 dimensions for the GP
        self.max_pol_emb_dim = 4
        # Policies are safely sampled 25 times (.5 seconds) each iteration.
        # Switching policies at a higher rate can cause stability issues.
        self.increment_samples = 25
        # After sampling, we discard the first 15 samples of each iteration when
        # fitting the GP. This is done to disregard the noise when switching policies
        self.noisy_samples = 15
        # If the final policy selected is different from the currently
        # loaded one, wait a few more timesteps before measuring policy
        # performance to stabilize the robot
        self.extra_timesteps = 15

        self.loadPolicies(obs_shape, act_shape)
        self.loadWorldModels()

        if self.policies.get(initial_pair):
            self.set_policy(initial_pair)

        print(f"Starting up with {initial_pair} world model & policy.")

        self.native_errors = {}
        if len(self.pol_names) > 0:
            self.inaffinity_matrix = jp.stack([self.computePolicyEmbedding(name) for name in self.pol_names])
            print(f"Raw inaffinity matrix:\n{self.inaffinity_matrix}")

        if len(self.pol_names) > 1:
            self.normalizePolicyEmbeddings()
        elif len(self.pol_names) == 1:
            self.set_policy(self.pol_names[0])

        # JIT compile the gradient descent logic
        self.fast_update = jax.jit(train_step)

    def export_history(self, path):
        data = {
            "inaffinity_matrix": self.inaffinity_matrix,
            "pol_names": self.pol_names,
            "detector": self.detector,
            "smooth_errors": self.smooth_errors,
            "policy_history": self.policy_history,
            "contact_history": self.contact_history,
            "drift_indices": self.drift_indices,
            "policy_embeddings": self.policy_embeddings,
            "env_changes": self.env_changes,
            "gp_states": self.gp_states
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def setEnv(self, env):
        self.env_changes.append((len(self.detector.stat_values), env.name))
        self.env = env

        if len(self.pol_names) == 0:
            self.adapt_policy(None)
            self.set_policy(self.pol_names[0])

    def normalizePolicyEmbeddings(self):
        if self.generatePlots:
            plots.policyEmbeddings3D(self)

        if len(self.pol_names) > self.max_pol_emb_dim:
            print(f"Applying SVD to reduce dimensionality from {len(self.pol_names)}D to 4D.")
            # Extract the top 4 principal components that explain the most variance
            reducer = TruncatedSVD(n_components=self.max_pol_emb_dim)
            inaffinity_matrix = reducer.fit_transform(self.inaffinity_matrix)

            # Convert back to JAX array for the rest of the pipeline
            inaffinity_matrix = jp.array(inaffinity_matrix)

            # Optional: Print how much of the original information was preserved
            variance_ratio = sum(reducer.explained_variance_ratio_) * 100
            print(f"SVD preserved {variance_ratio:.2f}% of the variance.")
        else:
            inaffinity_matrix = self.inaffinity_matrix

        # Normalization to get final embeddings.
        # After this step, we will be able to measure the cosine distance between them to get the "semantic"
        # difference between two different policies.
        norm_mat = inaffinity_matrix / jp.linalg.norm(inaffinity_matrix, axis=1, keepdims=True)

        self.policy_embeddings = {
            name: embedding for name, embedding in zip(self.pol_names, norm_mat)
        }

        print(f"Normalized policy embeddings:")
        for pol_name in self.policy_embeddings:
            print(f"{pol_name}: {self.policy_embeddings[pol_name]}")

    def loadWorldModels(self):
        self.wm_dict = {}

        for env_name in self.pol_names:
            # Load Pre-trained Weights
            with open(WM_PATH.format(env_name=env_name), 'rb') as f:
                params = pickle.load(f)
            with open(WM_STATS_PATH.format(env_name=env_name), 'rb') as f:
                stats = pickle.load(f)

            sensor_dim = stats['obs_mean'].shape[0]
            action_dim = stats['act_mean'].shape[0]

            # Initialize training state (Optimizer)
            rng = jax.random.PRNGKey(0)
            wm_state = create_train_state(rng, learning_rate=1e-5, sensor_dim=sensor_dim, action_dim=action_dim) # Low LR for stability
            wm_info = (env_name, wm_state.replace(params=params), stats)

            self.wms.append(wm_info)
            self.wm_dict[env_name] = wm_info

    def loadPolicies(self, obs_shape, act_shape):
        basePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

        # Load network parameters
        # They are the same for all environments
        self.ppo_params = locomotion_params.brax_ppo_config("Go2StrollFlatTerrain", IMPL)

        for env_name in self.pol_names:
            checkpoint_path = basePath + "/" + POL_PATH.format(env_name=env_name)

            normalize = lambda x, y: x
            if self.ppo_params.normalize_observations:
                normalize = running_statistics.normalize

            network_fn = (
                ppo_networks.make_ppo_networks
            )
            if hasattr(self.ppo_params, "network_factory"):
                network_factory = functools.partial(
                    network_fn, **self.ppo_params.network_factory
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


    def getPredictionWM(self, wm, stats, obs, action):
        # Predict what should have happened
        return _jit_predict_wm(wm.apply_fn, wm.params, stats, obs, action)

    def getErrorWM(self, name, wm, stats, obs, action, next_obs):
        pred_next_obs = self.getPredictionWM(wm, stats, obs, action)

        # Calculate error
        error = jp.linalg.norm(pred_next_obs - next_obs)
        alpha = 0.1
        smooth_error = error if len(self.errors[name]) == 0 else alpha * error + (1-alpha) * self.errors[name][-1]

        self.errors[name].append(error)
        self.smooth_errors[name].append(smooth_error)

        return error

    def computePolicyEmbedding(self, env_name):
        '''
        The embeddings of a policy are its affinity coefficients to each environment.
        '''

        datasetPath = WM_DS_PATH.format(env_name=env_name)

        # Load data
        obs_data, act_data, next_data = load_dataset(datasetPath)

        errors = jp.vstack([
            jp.linalg.norm(self.getPredictionWM(wm, stats, obs_data, act_data) - next_data, axis=1)
            for _, wm, stats in self.wms
        ])
        mean_errors = jp.mean(errors, axis=1)

        env_index = self.pol_names.index(env_name)
        # Compute the extra "surprise" by subtracting the baseline noise
        baseline_error = mean_errors[env_index]
        embedding = jp.abs(mean_errors - baseline_error)

        self.native_errors[env_name] = (baseline_error, errors[env_index,:])

        return embedding

    def predictPolicyScore(self, name, active_name):
        # GP predicts the expected error and uncertainty
        pol_emb = self.policy_embeddings[name]
        active_emb = self.policy_embeddings[active_name]
        mean, std = self.gp.predict([pol_emb], return_std=True)

        # UCB formula: Exploit (mean) + Explore (Kappa * std) * Safety constraints

        # It is dangerous to rollout policies that are too different together for stability reasons.
        # With the safety constraint, the score for transitioning from one to another is lower when the
        # policies have little in common.
        cos_dist = 1 - jp.dot(pol_emb, active_emb) / (jp.linalg.norm(pol_emb) * jp.linalg.norm(active_emb))
        kappa = 2.; gamma = .5
        ucb_score = -mean[0] + kappa * std[0] * jp.exp(-gamma * cos_dist)
        return ucb_score, mean, std

    def getFinalPolicyScore(self, pol_name):
        '''
        Computes the expected error from the policy's obtained errors.
        We use only the real data obtained for the final decision after a
        policy search process.
        '''
        # If no data was collected for this policy, the GP model did not
        # think it would be promissing. Donnot choose it.
        if len(self.sampled_errors[pol_name]) == 0:
            return -jp.inf

        samples = jp.array(self.sampled_errors[pol_name])

        remainder = len(samples) % self.increment_samples
        first_samples = samples[:remainder]
        samples = samples[remainder:].reshape([-1, self.increment_samples])[:,self.noisy_samples:].ravel()
        samples = jp.append(first_samples, samples)

        # Reward consistency by looking at the mean
        return -jp.mean(samples)

    def getNextPolicy(self, active_name):
        # Fit the GP with the new real-world data
        noise = 1e-06
        
        # Even with normalize_y=True the alpha vector has to be normalized
        # But when y_var is 0, scikit doesn't scale the data, so we divide by 1.0
        y_var = float(jp.var(self.y_train))
        y_var_safe = 1.0 if y_var == 0.0 else y_var
        normalized_alpha = (self.alpha + noise) / y_var_safe

        prior_variance = float(max(1.0, jp.max(normalized_alpha) * 2.0))

        max_dist = jp.sqrt(min(len(self.pol_names), self.max_pol_emb_dim))
        kernel = C(prior_variance, constant_value_bounds="fixed") * RBF(length_scale=max_dist/2, length_scale_bounds=(1e-2, max_dist))

        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=normalized_alpha)
        self.gp.fit(self.X_train, self.y_train)

        polInfo = [self.predictPolicyScore(name, active_name) + (name,) for name in self.policies]
        best_ucb, mean, std, next_name = max(polInfo)

        loggingInfo = [self.iteration, active_name, next_name, polInfo]
        if self.iteration == 1:
            self.gp_states.append([loggingInfo])
        else:
            self.gp_states[-1].append(loggingInfo)

        return next_name

    def adapt_policy(self, base_policy_name):
        raise NotImplementedError("Method implemented in OfflineRobotController")

    def set_policy(self, pol_name):
        self.active_wm = self.wm_dict[pol_name]
        self.inference = self.policies[pol_name]

    def control_loop(self, obs, action, state):
        if not self.deploy:
            return

        obs, next_obs = obs["wm_state"], state.obs["wm_state"]

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
                if self.iteration < self.max_iterations:
                    self.iteration += 1
                    next_name = self.getNextPolicy(prev_active_name)
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

                    print(f"Beginning iteration {self.iteration} with {next_name}.")
                elif self.iteration == self.max_iterations:
                    # To make the final decision, delegating on the GP to choose the best policy may be incorrect.
                    # The GP is curious and may want to explore a novel route after having seen a lot of data of the
                    # best performing policy, since the search is based on UCB scores.

                    # Thus, we base the final choice on scores computed only from the real data collected.
                    errs = [(self.getFinalPolicyScore(pol_name), pol_name) for pol_name in self.sampled_errors]
                    _, next_name = max(errs)

                    print(f"Converged on {next_name}.")
                    self.set_policy(next_name)

                    if prev_active_name != next_name:
                        self.iteration += 1
                        print("Waiting extra time after final switch")
                        self.samples2collect[next_name] += self.extra_timesteps
                    else:
                        self.sampling = False

                        if self.generatePlots:
                            plots.plotGaitPattern(self)
                            plots.plotGPSearch(self)
                else: # End of extra stability iteration
                    self.set_policy(prev_active_name)
                    self.sampling = False

                    if self.generatePlots:
                        plots.plotGaitPattern(self)
                        plots.plotGPSearch(self)

        active_name, wm, stats = self.active_wm
        error = self.getErrorWM(active_name, wm, stats, obs, action, next_obs)
        # Only needed for plotting WM error of all WMs
        [self.getErrorWM(name, wm, stats, obs, action, next_obs) for name, wm, stats in self.wms if name != active_name]
            
        # Fill out histories for plotting
        self.contact_history.append(state.info["last_contact"])
        self.policy_history.append(active_name)

        if self.sampling:
            # Needed for plotting, we append the 0 manually as we don't want to feed the dummy 0
            # into the KS-ADWIN detection pipeline
            self.detector.stat_values.append(0)
            self.sampled_errors[active_name].append(error)

            # When sampling batch is over for the policy, compute its new GP datapoint
            if len(self.sampled_errors[active_name]) == self.samples2collect[active_name]:
                samples = jp.array(self.sampled_errors[active_name])
                remainder = len(samples) % self.increment_samples
                first_samples = samples[:remainder]
                samples = samples[remainder:].reshape([-1, self.increment_samples])[:,self.noisy_samples:].ravel()
                samples = jp.append(first_samples, samples)

                self.X_train = np.vstack([self.X_train, self.policy_embeddings[active_name]])
                self.y_train = np.append(self.y_train, jp.mean(samples))
                self.alpha = np.append(self.alpha, jp.var(samples)/len(samples))

            return

        # Update detector
        is_drift, _, policy_performance_alert = self.detector.update(error, self.native_errors[active_name])

        if is_drift or policy_performance_alert:
            if policy_performance_alert:
                print(f"Policy performance alert")
            else:
                print(f"Domain change detected")

            # The only case where a performance alert is not a valid drift detection
            # happens when the actual drift detection module still doesn't have enough samples yet.
            # If this happens, it must be that a policy search process has just finished and the selected
            # policy is not good enough (results in serious gait instabilities). Thus, we should adapt a new policy
            # at this point.
            adapt = policy_performance_alert and not is_drift and len(self.drift_indices) > 0
            # Adaptation is also needed if there was a change alert and we only have 1 policy
            if len(self.pol_names) == 1 or adapt:
                try:
                    self.adapt_policy(active_name)
                    return
                except NotImplementedError:
                    if len(self.pol_names) == 1:
                        return
                    # If the adapt_policy method is not implemented, then we are deploying the robot
                    # with all needed policies loaded in-simulation or in the real world.
                    # Thus we have entered a failure mode, as the GP search did not find the optimal policy.
                    # We try again with the GP search in case it was bad luck.
                    pass

            self.sampled_errors = {name: [] for name, _, _ in self.wms}

            # We will "seed" the GP search process with some initial samples.
            # IF performance alert: sample only 2 current errors as sometimes these alerts are so quick that 2
            # consecutive, very big error samples are enough signal for it to fire.
            # IF not performance alert: we can assign last obtained errors so that we may start the search process
            # with a bit more useful data.
            init_samples = 2 if policy_performance_alert else self.increment_samples-1
            samples = self.errors[active_name][-init_samples:]
            self.sampled_errors[active_name] = samples

            self.X_train = np.array([self.policy_embeddings[active_name]])
            samples = jp.array(samples)
            self.y_train = np.array([jp.mean(samples)])
            self.alpha = np.array([jp.var(samples)/len(samples)])

            self.iteration = 1

            self.samples2collect = {active_name: len(samples)}
            next_name = self.getNextPolicy(active_name)

            # When rolling out different policies, the amount of samples to collect of each
            self.samples2collect[next_name] = self.increment_samples

            self.sampling = True

            self.drift_indices.append(len(self.detector.stat_values))

            # Plotting
            if self.generatePlots:
                plots.wmErrorHistory(self)
                plots.statisticDriftHistory(self)
