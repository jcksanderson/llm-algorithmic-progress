#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Define your training commands sequentially using '&&'
# Replace these with your actual torchrun commands and config files

echo "Starting Run 1 (Small Model, Config 1)"
torchrun --standalone --nproc_per_node=1 train.py config/compact/prototype_sparse_attn.py >> logs/run1.log 2>&1 && \
echo "Run 1 finished successfully." && \

echo "Starting Run 2 (Small Model, Config 2)"
torchrun --standalone --nproc_per_node=1 train.py config/compact/prototype_layer_norm.py >> logs/run2.log 2>&1 && \
echo "Run 2 finished successfully." && \

echo "All training runs completed."
