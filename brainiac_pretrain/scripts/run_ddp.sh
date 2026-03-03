#!/bin/bash
# Script to run distributed training with torchrun

# Configuration
CONFIG_PATH="configs/pretrain_simclr.yaml"
NUM_GPUS=4  # Adjust based on your setup
MASTER_PORT=29500

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
