# --- Enable Sparse Attention and Configure WandB ---
use_sparse_attn = True
native_sparse_causal = True   # Ensure this is True for GPT
use_rope = False              # RoPE is bypassed by this sparse module, so set to False

wandb_log = False 

# --- Dataset ---
dataset = 'openwebtext'

# --- Make the run short for testing ---
max_iters = 200          # Total training iterations
eval_interval = 50       # How often to run evaluation
eval_iters = 10          # Number of batches for each evaluation
log_interval = 10        # How often to print to console

# --- Reduce batch size and accumulation for faster iterations during test ---
batch_size = 4           # Micro-batch size
gradient_accumulation_steps = 1 # No gradient accumulation

# compile = False # Optional: Turn off compile for faster startup during initial tests
