# config/test_no_features.py
from config.full.train_gpt2 import *  # Import base config

# Override with specific configuration
wandb_log = True
wandb_project = 'llm-progress'
wandb_run_name = 'full - sparse_attn'

# Feature toggles
use_layer_norm = False
use_rope = False
use_flash_attn = False
use_sparse_attn = True
use_mqa = False

sparse_n_head = 64
sparse_kv_n_head = 4
