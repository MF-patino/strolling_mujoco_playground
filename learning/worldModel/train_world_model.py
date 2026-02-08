import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import os
import glob
import pickle

# Configuration
HIDDEN_DIMS = [512, 256, 128]
LEARNING_RATE = 1e-4
EARLY_STOP_EPOCHS = 40
BATCH_SIZE = 512
EPOCHS = 1000
ALL_ENVS = ["Go2StrollFlatTerrain", "Go2StrollRoughTerrain"]

# Define the predictive part of the World Model, the MLP
class WorldModelMLP(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, obs, action):
        # Concatenate State + Action
        x = jnp.concatenate([obs, action], axis=-1)
        
        # MLP Layers
        for size in HIDDEN_DIMS:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
            
        # Output Layer (Predicts DELTA, not next state)
        # It's easier to learn (Next - Cur) than Next
        delta = nn.Dense(self.output_dim)(x)
        return delta

# Helper to create a TrainState
def create_train_state(rng, learning_rate, sensor_dim, action_dim):
    """Creates initial model parameters and optimizer state."""
    model = WorldModelMLP(output_dim=sensor_dim)
    
    # Initialize dummy inputs to infer shapes
    dummy_obs = jnp.ones((1, sensor_dim))
    dummy_act = jnp.ones((1, action_dim))
    params = model.init(rng, dummy_obs, dummy_act)['params']
    
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

# Training Step (JIT Compiled)
@jax.jit
def train_step(state, batch_obs, batch_act, batch_next_delta):
    """
    Performs one step of Gradient Descent.
    This is EXACTLY what you will run on the robot for online learning.
    """
    
    def loss_fn(params):
        # Predict delta
        pred_delta = state.apply_fn({'params': params}, batch_obs, batch_act)
        
        # MSE Loss
        loss = jnp.mean((pred_delta - batch_next_delta) ** 2)
        return loss

    # Compute Gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Update Weights
    state = state.apply_gradients(grads=grads)
    return state, loss

# Function for evaluation, same logic as loss_fn
@jax.jit
def eval_step(state, batch_obs, batch_act, batch_next_obs):
    pred_delta = state.apply_fn({'params': state.params}, batch_obs, batch_act)
    return jnp.mean((pred_delta - batch_next_obs) ** 2)

# Data loading
def load_dataset(datasetPath):
    print(f"Loading data from {datasetPath}...")
    files = glob.glob(os.path.join(datasetPath, "*.npz"))
    
    all_obs, all_act, all_next = [], [], []
    
    for f in files:
        data = np.load(f)
        all_obs.append(data["obs"])
        all_act.append(data["action"])
        all_next.append(data["next_obs"])
        
    # Concatenate
    X_obs = np.concatenate(all_obs, axis=0)
    X_act = np.concatenate(all_act, axis=0)
    Y_next = np.concatenate(all_next, axis=0)
    
    print(f"Total Dataset Size: {X_obs.shape[0]}")
    return X_obs, X_act, Y_next
    
# Main training loop
def trainWM(env_name):
    root = f"world_models/{env_name}/"
    datasetPath = root + "world_model_dataset"
    modelPath = root + "world_model_best.pkl"
    statsPath = root + "normalization_stats.pkl"

    print("\n________________________________________________")
    print(f"Training World Model for {env_name} environment")
    # Load data
    obs_data, act_data, next_data = load_dataset(datasetPath)
    delta_data = next_data - obs_data
    num_samples = obs_data.shape[0]
    
    # Initialize model
    rng = jax.random.PRNGKey(0)
    sensor_dim, action_dim = obs_data.shape[1], act_data.shape[1]
    state = create_train_state(rng, LEARNING_RATE, sensor_dim, action_dim)

    # Train/validation division

    split_idx = int(num_samples * 0.8)
    indices = np.random.permutation(num_samples)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    print("Preprocessing...")
    # Compute stats on training set
    obs_mean = np.mean(obs_data[train_idx], axis=0)
    obs_std = np.std(obs_data[train_idx], axis=0)
    
    act_mean = np.mean(act_data[train_idx], axis=0)
    act_std = np.std(act_data[train_idx], axis=0)
    
    delta_mean = np.mean(delta_data[train_idx], axis=0)
    delta_std = np.std(delta_data[train_idx], axis=0)

    # Apply normalization to all data
    obs = (obs_data - obs_mean) / obs_std
    act = (act_data - act_mean) / act_std
    target = (delta_data - delta_mean) / delta_std

    # Create train/val datasets
    train_set = (obs[train_idx], act[train_idx], target[train_idx])
    val_set = (obs[val_idx], act[val_idx], target[val_idx])

    # Save statistics for the robot
    with open(statsPath, "wb") as f:
        pickle.dump({
            "obs_mean": obs_mean, "obs_std": obs_std,
            "act_mean": act_mean, "act_std": act_std,
            "delta_mean": delta_mean, "delta_std": delta_std
        }, f)

    print("Starting training...")
    
    # Training loop
    best_val_loss = float('inf')
    worse_epochs = 0
    for epoch in range(EPOCHS):
        # Shuffle
        perm = np.random.permutation(len(train_idx))
        train_losses = []
        val_losses = []
        
        # Perform a training epoch
        for i in range(0, len(train_idx), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            b_obs, b_act, b_target = train_set[0][idx], train_set[1][idx], train_set[2][idx]
            
            state, loss = train_step(state, b_obs, b_act, b_target)
            train_losses.append(loss)
        
        # Perform a validation epoch
        for i in range(0, len(val_idx), BATCH_SIZE):
            b_obs, b_act, b_target = val_set[0][i:i+BATCH_SIZE], val_set[1][i:i+BATCH_SIZE], val_set[2][i:i+BATCH_SIZE]
            loss = eval_step(state, b_obs, b_act, b_target)
            val_losses.append(loss)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        print(f"Epoch {epoch} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        # Checkpointing (based on early-stopping)
        if avg_val < best_val_loss:
            worse_epochs = 0
            best_val_loss = avg_val
            with open(modelPath, "wb") as f:
                pickle.dump(state.params, f)
        elif worse_epochs >= EARLY_STOP_EPOCHS:
            break
        else:
            worse_epochs += 1
def main():
    # All world models are trained from scratch
    for env in ALL_ENVS:
        trainWM(env)
    

if __name__ == "__main__":
    main()