# config/test_mqa.py
from config.compact.train_gpt2_compact import *  # Import base config

# Override with specific configuration
wandb_log = True
wandb_project = 'llm-progress'
wandb_run_name = 'compact - flash_attn + rope + layer_norm'

# Feature toggles
use_layer_norm = True
use_rope = True
use_flash_attn = True
use_sparse_attn = False
use_mqa = False 
