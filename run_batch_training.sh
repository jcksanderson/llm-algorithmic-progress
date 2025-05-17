#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# HACK: ========== Compact Model Training Runs ==========

echo "--- Starting Batch Training Run (Compact Configs) ---"

# Run 1: test_no_features.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 1 - Compact, No Features"
torchrun --standalone --nproc_per_node=4 train.py config/compact/test_no_features.py >> logs/compact/run1_compact_no_features.log 2>&1 && \
echo "Run 1 finished successfully." && \

# Run 2: test_flash_attn.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 2 - Compact, Flash Attention"
torchrun --standalone --nproc_per_node=4 train.py config/compact/test_flash_attn.py >> logs/compact/run2_compact_flash_attn.log 2>&1 && \
echo "Run 2 finished successfully." && \

# Run 3: test_layer_norm.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 3 - Compact, Layer Norm"
torchrun --standalone --nproc_per_node=4 train.py config/compact/test_layer_norm.py >> logs/compact/run3_compact_layer_norm.log 2>&1 && \
echo "Run 3 finished successfully." && \

# Run 4: test_rope.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 4 - Compact, RoPE"
torchrun --standalone --nproc_per_node=4 train.py config/compact/test_rope.py >> logs/compact/run4_compact_rope.log 2>&1 && \
echo "Run 4 finished successfully." && \

# Run 5: test_mqa.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 5 - Compact, MQA"
torchrun --standalone --nproc_per_node=4 train.py config/compact/test_mqa.py >> logs/compact/run5_compact_mqa.log 2>&1 && \
echo "Run 5 finished successfully." && \

# Run 6: test_sparse_attn.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 6 - Compact, Sparse Attention"
torchrun --standalone --nproc_per_node=4 train.py config/compact/test_sparse_attn.py >> logs/compact/run6_compact_sparse_attn.log 2>&1 && \
echo "Run 6 finished successfully." && \

# Run 7: test_all.py (Assuming this means all features enabled in its config)
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 7 - Compact, All Features"
torchrun --standalone --nproc_per_node=4 train.py config/compact/test_all.py >> logs/compact/run7_compact_all_features.log 2>&1

echo "--- Compact Batch Training Runs Completed Successfully ---"


# HACK: ========== Full Model Training Runs ========== 

echo "--- Starting Batch Training Run (Full Configs) ---"

# Run 1: test_no_features.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 1 - Full, No Features"
torchrun --standalone --nproc_per_node=4 train.py config/full/test_no_features.py >> logs/full/run1_full_no_features.log 2>&1 && \
echo "Run 1 finished successfully." && \

# Run 2: test_flash_attn.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 2 - Full, Flash Attention"
torchrun --standalone --nproc_per_node=4 train.py config/full/test_flash_attn.py >> logs/full/run2_full_flash_attn.log 2>&1 && \
echo "Run 2 finished successfully." && \

# Run 3: test_layer_norm.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 3 - Full, Layer Norm"
torchrun --standalone --nproc_per_node=4 train.py config/full/test_layer_norm.py >> logs/full/run3_full_layer_norm.log 2>&1 && \
echo "Run 3 finished successfully." && \

# Run 4: test_rope.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 4 - Full, RoPE"
torchrun --standalone --nproc_per_node=4 train.py config/full/test_rope.py >> logs/full/run4_full_rope.log 2>&1 && \
echo "Run 4 finished successfully." && \

# Run 5: test_mqa.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 5 - Full, MQA"
torchrun --standalone --nproc_per_node=4 train.py config/full/test_mqa.py >> logs/full/run5_full_mqa.log 2>&1 && \
echo "Run 5 finished successfully." && \

# Run 6: test_sparse_attn.py
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 6 - Full, Sparse Attention"
torchrun --standalone --nproc_per_node=4 train.py config/full/test_sparse_attn.py >> logs/full/run6_full_sparse_attn.log 2>&1 && \
echo "Run 6 finished successfully." && \

# Run 7: test_all.py (Assuming this means all features enabled in its config)
echo "[$(date "+%m-%d %H:%M:%S")] Starting Run 7 - Full, All Features"
torchrun --standalone --nproc_per_node=4 train.py config/full/test_all.py >> logs/full/run7_full_all_features.log 2>&1

echo "--- Full Batch Training Runs Completed Successfully ---"
echo "--- All Training Successful ---"
