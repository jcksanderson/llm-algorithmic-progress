# config/test_mqa.py
from config.train_gpt2_compact import *  # Import base config

# Override with specific configuration
wandb_log = True
wandb_project = 'nanoGPT-algo-test'
wandb_run_name = 'flash_attn + rope + layer_norm'

# Feature toggles
use_layer_norm = True
use_rope = True
use_flash_attn = True
use_sparse_attn = False
use_mqa = False 