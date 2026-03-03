"""
Sampler utilities for distributed training.
"""
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler


def build_sampler(
    dataset,
    shuffle: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
):
    """
    Build appropriate sampler for training.
    
    Args:
        dataset: Dataset instance
        shuffle: Whether to shuffle
        distributed: Whether using DDP
        rank: Process rank (for DDP)
        world_size: Total number of processes (for DDP)
        seed: Random seed
    
    Returns:
        Sampler instance
    """
    if distributed:
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        if shuffle:
            return RandomSampler(dataset)
        else:
            return SequentialSampler(dataset)
