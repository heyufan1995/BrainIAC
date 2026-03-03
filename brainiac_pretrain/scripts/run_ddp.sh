#!/bin/bash
# Script to run distributed training with torchrun

# Configuration
CONFIG_PATH="configs/pretrain_simclr.yaml"
# Auto-detect number of GPUs, or set manually (e.g., NUM_GPUS=8)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
MASTER_PORT=29500

echo "Using $NUM_GPUS GPUs for training"

# Run training
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config ${CONFIG_PATH}

# Alternative: Use python -m torch.distributed.launch (older PyTorch)
# python -m torch.distributed.launch \
#     --nproc_per_node=${NUM_GPUS} \
#     --master_port=${MASTER_PORT} \
#     train.py \
#     --config ${CONFIG_PATH}
