# Test configuration for compact GPT-2 - Minimal training to validate setup
# Use: python train.py config/test_gpt2_compact.py

# Import base compact configuration
from config.train_gpt2_compact import *

# Override for quick testing
max_iters = 50        # Just 50 iterations to test setup
eval_interval = 10    # Evaluate more frequently
eval_iters = 5       # Use fewer iterations for evaluation
log_interval = 1     # Log every iteration

# Reduce batch size for quicker iteration
batch_size = 4
gradient_accumulation_steps = 2

# Dataset
dataset = 'shakespeare_char'  # Use character-level Shakespeare dataset
wandb_log = True             # Enable wandb logging
wandb_project = 'nanogpt'    # Your project name
wandb_run_name = 'compact-gpt2-test'  # Name for this particular run
