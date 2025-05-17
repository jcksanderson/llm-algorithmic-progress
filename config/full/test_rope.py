# config/test_no_features.py
from config.full.train_gpt2 import *  # Import base config

# Override with specific configuration
wandb_log = True
wandb_project = 'llm-progress'
wandb_run_name = 'full - rope'

# Feature toggles
use_layer_norm = False
use_rope = True
use_flash_attn = False
use_sparse_attn = False
use_mqa = False
