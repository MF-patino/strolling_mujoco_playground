import os
import time

import jax
import jax.numpy as jp
import numpy as np

class WorldModelRolloutSaver:
    """Collect world-model transitions each time there is a rollout.

    Designed to be called from PPO's policy_params_fn (same hook as rscope).
    """

    def __init__(
        self,
        env,
        episode_length: int,
        num_envs: int = 64,
        data_dir: str = "world_model_dataset",
        deterministic: bool = True,
    ):
        self.env = env
        self.episode_length = episode_length
        self.num_envs = num_envs
        self.deterministic = deterministic
        self._make_policy = None
        self._last_collect_step = -1

        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

        # Compile once and reuse
        self._compiled_rollout = None

    def set_make_policy(self, make_policy_fn):
        self._make_policy = make_policy_fn
        # invalidate compiled rollout if policy factory changes
        self._compiled_rollout = None

    def dump_rollout(self, params, current_step):
        """
        Runs rolled-out policy in eval environments on the GPU efficiently and 
        then writes the state transitions to disk.
        """

        # Build inference function
        inference_fn = self._make_policy(
            params, deterministic=self.deterministic
        )

        # Compile rollout function lazily (once)
        if self._compiled_rollout is None:
            self._compiled_rollout = self._build_rollout_fn(inference_fn)

        # RNG for parallel envs
        rng = jax.random.PRNGKey(int(time.time()) & 0xFFFFFFFF)
        rngs = jax.random.split(rng, self.num_envs)

        # Reset environments
        reset_states = jax.jit(jax.vmap(self.env.reset))(rngs)

        # Run rollouts on device
        traj_obs, traj_action, traj_next_obs = self._compiled_rollout(
            rngs, reset_states
        )

        # Get trajectories from GPU to RAM
        obs, action, next_obs = jax.device_get(
            (traj_obs, traj_action, traj_next_obs)
        )

        # Flatten (T, N, ...) -> (T*N, ...)
        T, N = obs.shape[:2]
        obs = obs.reshape((T * N,) + obs.shape[2:])
        action = action.reshape((T * N,) + action.shape[2:])
        next_obs = next_obs.reshape((T * N,) + next_obs.shape[2:])

        # Write the whole chunk synchronously
        path = os.path.join(self.data_dir, f"wm_chunk_{current_step}.npz")

        np.savez_compressed(
            path,
            obs=obs,
            action=action,
            next_obs=next_obs,
        )

        print(
            f"[WorldModelSaver] Saved {obs.shape[0]} transitions to {path}"
        )

    def _build_rollout_fn(self, inference_fn):
        """Create and JIT compile a rollout function for the current policy."""

        def _single_env_rollout(rng, state):
            def _step(carry, _):
                st, rng = carry
                rng, subkey = jax.random.split(rng)
                action = inference_fn(st.obs, subkey)[0]
                next_st = self.env.step(st, action)
                # state representation comes from a special entry called "wm_state"
                record = (st.obs["wm_state"], action, next_st.obs["wm_state"])
                return (next_st, rng), record

            (_, _), traj = jax.lax.scan(
                _step, (state, rng), None, length=self.episode_length
            )
            return traj

        # Vectorize over envs
        batched = jax.vmap(_single_env_rollout, in_axes=(0, 0))

        # JIT once
        return jax.jit(batched)