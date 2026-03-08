MODELS_ROOT = "model_pairs/{env_name}/"
WM_ROOT = MODELS_ROOT + "world_model/"

WM_DS_PATH = WM_ROOT + "world_model_dataset"
WM_PATH = WM_ROOT + "world_model_best.pkl"
WM_STATS_PATH = WM_ROOT + "normalization_stats.pkl"

POL_PATH = MODELS_ROOT + "policy"

ALL_ENVS = ["Go2StrollSlipperyTerrain", "Go2StrollFlatTerrain", "Go2StrollRoughTerrain"]