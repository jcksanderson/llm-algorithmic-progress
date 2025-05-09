# config for training compact GPT-2 (~50M) - optimized for faster training
# launch as: $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_compact.py

# wandb logging setup
wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-50M-h100-fast'

# Model architecture changes
# Total params reduced from 124M to ~50M
n_layer = 8
n_head = 8
n_embd = 512
block_size = 512
dropout = 0.1  # Slightly reduced for faster training

#Toggling algorithmic improvements:
use_layer_norm = False
use_rope = False
use_flash_attn = False
use_sparse_attn = False
use_mqa = False 

# Batch size configuration
# Original: 12 * 1024 * 40 = 491,520
# New: 64 * 512 * 8 = 262,144 (still substantial)
batch_size = 64
gradient_accumulation_steps = 8

# Training schedule - reduced for faster results
max_iters = 50000  
warmup_iters = 1000
lr_decay_iters = 50000

# Learning rate configuration
# Scaled based on model size: 6e-4 * sqrt(512/768)
learning_rate = 8e-4  # Increased from 6e-4
min_lr = 8e-5
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
lr_decay_iters = 50000  # Reduced to match max_iters
decay_lr = True
lr_decay_type = 'cosine'  # Keep this parameter

# Evaluation settings
eval_interval = 500
eval_iters = 100
log_interval = 10

# Regularization
weight_decay = 1e-1
