from config.full.train_gpt2 import *  # Import base config

# Feature toggles
use_layer_norm = False
use_rope = False
use_flash_attn = False
use_sparse_attn = True
use_mqa = False

# no logging
wandb_log = False 

# --- Make the run short for testing ---
max_iters = 200          # Total training iterations
eval_interval = 50       # How often to run evaluation
eval_iters = 10          # Number of batches for each evaluation
log_interval = 10        # How often to print to console

# --- Reduce batch size and accumulation for faster iterations during test ---
batch_size = 4
gradient_accumulation_steps = 4

