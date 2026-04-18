# Online policy adaptation with Mujoco Playground 

This fork of Mujoco Playground provides a general framework for online adaptation of Go2 quadruped gaits in environments with changing terrain geometries and friction coefficients. The proposed solution bypasses catastrophic forgetting completely by building a scaffold that forever catalogs policies and also efficiently retrieves the best one (among hundreds or thousands) for a given environment.

## Instalation instructions

* Clone this repository
* Create a Python virtual environment: source ~/.venv/bin/activate
    ```sh 
    source ~/.venv/bin/activate
    ```
* Install the river online learning library: 
    - Install Rust: 
        ```sh 
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        ```
    - Choose option 1
    - Close the terminal and open a new one, then execute: 
        ```sh 
        pip install river
        ```

* (Optional) Install rscope:
    ```sh 
    pip install rscope
    ```

## Usage

### For deployment and visualization

* For deploying the online policy-WM pair adaptation stack:
    ```sh 
    python learning/visualize_adaptation.py
    ```
* Visualizing plots: 
    ```sh 
    python learning/plot_graphs.py
    ```
* Visualizing a policy in its native environment: 
    ```sh 
    python learning/train_jax_ppo.py --env_name [env_name] --play_only=True --load_checkpoint_path [checkpoint_path]
    ```

### For training world models offline

This script trains all world models (one for each environment) at the same time, using the offline datasets:

```sh 
python learning/worldModel/train_world_model.py
```

### For training policies

The usage command differ in the case of using the library rscope to visualize training results or not.

Commands without rscope:
* Training a policy from scratch: 
    ```sh 
    python learning/train_jax_ppo.py --env_name [env_name] --run_evals=False
    ```

* Training a policy from another one (for checkpointing or transfer learning): 
    ```sh 
    python learning/train_jax_ppo.py --env_name [env_name] --load_checkpoint_path [checkpoint_path]
    ```

Commands with rscope:
* Training a policy from scratch: 
    ```sh 
    python learning/train_jax_ppo.py --env_name [env_name] --rscope_envs 16 --run_evals=False --deterministic_rscope=True
    ```
* Training a policy from another one (for checkpointing or transfer learning): 
    ```sh 
    python learning/train_jax_ppo.py --env_name [env_name] --rscope_envs 16 --run_evals=False --deterministic_rscope=True  --load_checkpoint_path [checkpoint_path]
    ```

* Visualizing intermediate training results with rscope: 
    ```sh 
    python -m rscope
    ```

## Policy-WM pairs

The online adaptation process uses pairs of policies and their respective world models to track their performance. These pairs can be found in the `model_pairs` directory.

If the online adaptation system finds an environment it does not recognize, it then automatically trains and generates a new policy-WM pair for it in this folder. These adapted pairs contain the keyword "AdaptedFrom" in their folder names.

## Playground-trained policies (checkpoint paths)

The `logs` folder contains policies trained using the Playground `learning/train_jax_ppo.py` tool. These policies were trained for testing purposes and can be used by the online adaptation system, or visualized using the aforementioned script.

Policies trained from scratch:
* Go2StrollFlatTerrain:
```[path_to_playground]/logs/Go2StrollFlatTerrain-20260307-205501/checkpoints```

* Go2StrollRoughTerrain:
```[path_to_playground]/logs/Go2StrollRoughTerrain-20260301-122953/checkpoints```

* Go2StrollSlipperyTerrain:
```[path_to_playground]/logs/Go2StrollSlipperyTerrain-20260217-133106/checkpoints```

Policies trained from Go2StrollFlatTerrain:
* Go2StrollRoughTerrain:
```[path_to_playground]/logs/Go2StrollRoughTerrain-20260308-131625/checkpoints```
