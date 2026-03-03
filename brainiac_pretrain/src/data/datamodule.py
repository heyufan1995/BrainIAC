"""
PyTorch Lightning DataModule for pretraining.
"""
from typing import Optional, Dict, Any, List
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import PretrainDataset
from .transforms import (
    build_base_pretrain_transform,
    build_patch_sampler_transform,
    build_view_augment_transform,
    TwoCropsTransform,
)


def simclr_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for SimCLR training.
    Handles both single-crop and multi-crop cases.
    
    Args:
        batch: List of dicts with "id", "view1", "view2" keys
    
    Returns:
        Batched dict with "id", "view1", "view2" tensors
    """
    ids = [item["id"] for item in batch]
    view1_list = [item["view1"] for item in batch]
    view2_list = [item["view2"] for item in batch]
    
    # Check if multi-crop (first view has K dimension)
    if view1_list[0].dim() == 5:  # (K, C, D, H, W)
        # Multi-crop: stack along batch dimension, keep K dimension
        # Each item is (K, C, D, H, W), stack to (B, K, C, D, H, W)
        view1 = torch.stack(view1_list, dim=0)
        view2 = torch.stack(view2_list, dim=0)
    else:  # (C, D, H, W)
        # Single-crop: stack normally
        view1 = torch.stack(view1_list, dim=0)  # (B, C, D, H, W)
        view2 = torch.stack(view2_list, dim=0)
    
    return {
        "id": ids,
        "view1": view1,
        "view2": view2,
    }


class PretrainDataModule(pl.LightningDataModule):
    """
    DataModule for SimCLR pretraining.
    """
    
    def __init__(
        self,
        json_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        # Base preprocessing
        normalize_nonzero: bool = True,
        normalize_channel_wise: bool = True,
        orientation: Optional[str] = "RAS",
        spacing: Optional[tuple] = None,
        # Patch sampling
        patch_roi: tuple = (96, 96, 96),
        resize_to: Optional[tuple] = None,
        num_crops_per_scan: int = 1,
        # Augmentation
        flip_prob: float = 0.5,
        flip_axes: Optional[list] = None,
        affine_prob: float = 0.5,
        rotate_range: tuple = (0.1, 0.1, 0.1),
        translate_range: tuple = (5, 5, 5),
        scale_range: tuple = (0.1, 0.1, 0.1),
        noise_prob: float = 0.2,
        noise_std: float = 0.05,
        blur_prob: float = 0.2,
        blur_sigma: tuple = (0.5, 1.0),
        contrast_prob: float = 0.2,
        contrast_gamma: tuple = (0.7, 1.3),
        scale_intensity_prob: float = 0.1,
        scale_intensity_factors: tuple = (0.8, 1.2),
        shift_intensity_prob: float = 0.1,
        shift_intensity_offset: tuple = (-0.1, 0.1),
    ):
        super().__init__()
        self.json_path = json_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Store transform config
        self.normalize_nonzero = normalize_nonzero
        self.normalize_channel_wise = normalize_channel_wise
        self.orientation = orientation
        self.spacing = spacing
        self.patch_roi = patch_roi
        self.resize_to = resize_to
        self.num_crops_per_scan = num_crops_per_scan
        self.flip_prob = flip_prob
        self.flip_axes = flip_axes
        self.affine_prob = affine_prob
        self.rotate_range = rotate_range
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.contrast_prob = contrast_prob
        self.contrast_gamma = contrast_gamma
        self.scale_intensity_prob = scale_intensity_prob
        self.scale_intensity_factors = scale_intensity_factors
        self.shift_intensity_prob = shift_intensity_prob
        self.shift_intensity_offset = shift_intensity_offset
        
        self.dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        # Build transforms
        base_transform = build_base_pretrain_transform(
            normalize_nonzero=self.normalize_nonzero,
            normalize_channel_wise=self.normalize_channel_wise,
            orientation=self.orientation,
            spacing=self.spacing,
        )
        
        patch_transform = build_patch_sampler_transform(
            patch_roi=self.patch_roi,
            resize_to=self.resize_to,
        )
        
        # Determine final size (should match model input size)
        # Use resize_to if specified, otherwise use patch_roi
        final_size = self.resize_to if self.resize_to is not None else self.patch_roi
        
        aug_transform = build_view_augment_transform(
            flip_prob=self.flip_prob,
            flip_axes=self.flip_axes,
            affine_prob=self.affine_prob,
            rotate_range=self.rotate_range,
            translate_range=self.translate_range,
            scale_range=self.scale_range,
            noise_prob=self.noise_prob,
            noise_std=self.noise_std,
            blur_prob=self.blur_prob,
            blur_sigma=self.blur_sigma,
            contrast_prob=self.contrast_prob,
            contrast_gamma=self.contrast_gamma,
            scale_intensity_prob=self.scale_intensity_prob,
            scale_intensity_factors=self.scale_intensity_factors,
            shift_intensity_prob=self.shift_intensity_prob,
            shift_intensity_offset=self.shift_intensity_offset,
            final_size=final_size,  # Ensure exact size after augmentations
        )
        
        # Combine into TwoCropsTransform
        full_transform = TwoCropsTransform(
            base_transform=base_transform,
            patch_transform=patch_transform,
            aug_transform=aug_transform,
            num_crops_per_scan=self.num_crops_per_scan,
        )
        
        # Create dataset
        self.dataset = PretrainDataset(
            json_path=self.json_path,
            transform=full_transform,
        )
    
    def train_dataloader(self):
        """Create training DataLoader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Use DistributedSampler if DDP
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # Important for DDP
            collate_fn=simclr_collate_fn,
        )
    
    def val_dataloader(self):
        """Validation dataloader (optional, can use same as train for SSL)."""
        return self.train_dataloader()
