#!/usr/bin/env python3
"""
Main training script for BrainIAC SimCLR pretraining.

Usage:
    # Single GPU
    python train.py --config configs/pretrain_simclr.yaml
    
    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train.py --config configs/pretrain_simclr.yaml
    
    # Resume from checkpoint
    python train.py --config configs/pretrain_simclr.yaml --resume_from checkpoints/brainiac-epoch=50.ckpt
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train.train_simclr import train_simclr


def main():
    parser = argparse.ArgumentParser(description="Train BrainIAC SimCLR model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Run training
    train_simclr(
        config_path=args.config,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
