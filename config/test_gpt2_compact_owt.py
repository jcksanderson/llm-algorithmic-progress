# Test configuration for compact GPT-2 with OpenWebText
# Minimal training to validate setup
# Use: python train.py config/test_gpt2_compact_owt.py

# Import base compact configuration
from config.train_gpt2_compact import *

# Override for quick testing
max_iters = 100       # Very short training
eval_interval = 20    # Evaluate more frequently
eval_iters = 5       # Use fewer iterations for evaluation
log_interval = 1      # Log every iteration

# Reduce batch size for quicker iteration
batch_size = 8
gradient_accumulation_steps = 4

# Dataset
dataset = 'openwebtext'      # Use OpenWebText dataset
wandb_project = 'nanogpt'    # Project name
wandb_run_name = 'compact-gpt2-owt-test'  # Name this test run
