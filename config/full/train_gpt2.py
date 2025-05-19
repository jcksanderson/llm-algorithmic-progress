# config for training full GPT-2 (~124M)

# Model architecture changes
# Total params ~124M
n_layer = 10
n_head = 12
n_embd = 768
block_size = 1024
dropout = 0.4  # for faster training

# Toggling algorithmic improvements:
use_layer_norm = False
use_rope = False
use_flash_attn = False
use_sparse_attn = False
use_mqa = False 

# Batch size configuration
batch_size = 64
gradient_accumulation_steps = 8

# Training schedule - reduced for faster results
max_iters = 50000  
warmup_iters = 1000
lr_decay_iters = 50000

# Learning rate configuration
# Scaled based on model size: 6e-4 * sqrt(512/768)
learning_rate = 3e-4  
min_lr = 3e-5
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
log_interval = 50

# Regularization
weight_decay = 1e-1
