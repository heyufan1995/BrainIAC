"""
Utilities for distributed data parallel (DDP) training.
"""
import os
import torch
import torch.distributed as dist
from typing import Optional


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
):
    """
    Initialize distributed training.
    
    Args:
        backend: Distributed backend (nccl for GPU, gloo for CPU)
        init_method: Initialization method (default: env://)
    """
    if init_method is None:
        init_method = "env://"
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is main (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
