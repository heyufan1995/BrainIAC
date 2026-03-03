"""
Transform pipelines for BrainIAC pretraining.
Implements base preprocessing, patch sampling, and view augmentation.
"""
import copy
from typing import Dict, Any, Optional, Tuple
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    NormalizeIntensityd,
    RandSpatialCropd,
    Resized,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
    EnsureTyped,
)


def build_base_pretrain_transform(
    normalize_nonzero: bool = True,
    normalize_channel_wise: bool = True,
    orientation: Optional[str] = "RAS",
    spacing: Optional[Tuple[float, float, float]] = None,
    dtype: str = "float32",
) -> Compose:
    """
    Build base preprocessing transform pipeline.
    
    Args:
        normalize_nonzero: Normalize only nonzero voxels
        normalize_channel_wise: Normalize each channel independently
        orientation: Target orientation (e.g., "RAS"). None to skip.
        spacing: Target spacing in mm (e.g., (1.0, 1.0, 1.0)). None to skip.
        dtype: Output dtype
    
    Returns:
        MONAI Compose transform
    """
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ]
    
    if orientation is not None:
        transforms.append(Orientationd(keys=["image"], axcodes=orientation))
    
    if spacing is not None:
        transforms.append(
            Spacingd(
                keys=["image"],
                pixdim=spacing,
                mode="bilinear",
            )
        )
    
    # Crop to foreground (brain region)
    transforms.append(
        CropForegroundd(
            keys=["image"],
            source_key="image",
        )
    )
    
    # Normalize intensity
    transforms.append(
        NormalizeIntensityd(
            keys=["image"],
            nonzero=normalize_nonzero,
            channel_wise=normalize_channel_wise,
        )
    )
    
    # Ensure type
    transforms.append(EnsureTyped(keys=["image"], dtype=dtype))
    
    return Compose(transforms)


def build_patch_sampler_transform(
    patch_roi: Tuple[int, int, int] = (96, 96, 96),
    resize_to: Optional[Tuple[int, int, int]] = None,
) -> Compose:
    """
    Build transform that samples a random patch from the foreground-cropped volume.
    
    Args:
        patch_roi: Size of patch to crop (before resize)
        resize_to: Target size after crop. If None, uses patch_roi.
    
    Returns:
        MONAI Compose transform
    """
    transforms = [
        RandSpatialCropd(
            keys=["image"],
            roi_size=patch_roi,
            random_size=False,
        )
    ]
    
    if resize_to is not None and resize_to != patch_roi:
        transforms.append(
            Resized(
                keys=["image"],
                spatial_size=resize_to,
                mode="trilinear",
            )
        )
    
    return Compose(transforms)


def build_view_augment_transform(
    flip_prob: float = 0.5,
    flip_axes: Optional[list] = None,
    affine_prob: float = 0.5,
    rotate_range: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    translate_range: Tuple[int, int, int] = (5, 5, 5),
    scale_range: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    noise_prob: float = 0.2,
    noise_std: float = 0.05,
    blur_prob: float = 0.2,
    blur_sigma: Tuple[float, float] = (0.5, 1.0),
    contrast_prob: float = 0.2,
    contrast_gamma: Tuple[float, float] = (0.7, 1.3),
    scale_intensity_prob: float = 0.1,
    scale_intensity_factors: Tuple[float, float] = (0.8, 1.2),
    shift_intensity_prob: float = 0.1,
    shift_intensity_offset: Tuple[float, float] = (-0.1, 0.1),
) -> Compose:
    """
    Build stochastic augmentation transform for creating views.
    
    Args:
        flip_prob: Probability of flipping
        flip_axes: Axes to flip (e.g., [2] for left-right). None uses all axes.
        affine_prob: Probability of applying affine transform
        rotate_range: Rotation range in radians
        translate_range: Translation range in voxels
        scale_range: Scale range
        noise_prob: Probability of adding Gaussian noise
        noise_std: Standard deviation of noise
        blur_prob: Probability of Gaussian blur
        blur_sigma: Blur sigma range
        contrast_prob: Probability of contrast adjustment
        contrast_gamma: Gamma range for contrast
        scale_intensity_prob: Probability of intensity scaling
        scale_intensity_factors: Intensity scale factors
        shift_intensity_prob: Probability of intensity shifting
        shift_intensity_offset: Intensity shift offset range
    
    Returns:
        MONAI Compose transform
    """
    transforms = []
    
    # Affine transforms (rotation, translation, scaling)
    if affine_prob > 0:
        transforms.append(
            RandAffined(
                keys=["image"],
                prob=affine_prob,
                rotate_range=rotate_range,
                translate_range=translate_range,
                scale_range=scale_range,
                padding_mode="border",
                mode="bilinear",
            )
        )
    
    # Flipping
    if flip_prob > 0:
        if flip_axes is None:
            flip_axes = [0, 1, 2]
        transforms.append(
            RandFlipd(
                keys=["image"],
                prob=flip_prob,
                spatial_axis=flip_axes,
            )
        )
    
    # Gaussian blur
    if blur_prob > 0:
        transforms.append(
            RandGaussianSmoothd(
                keys=["image"],
                prob=blur_prob,
                sigma_x=blur_sigma,
                sigma_y=blur_sigma,
                sigma_z=blur_sigma,
            )
        )
    
    # Gaussian noise
    if noise_prob > 0:
        transforms.append(
            RandGaussianNoised(
                keys=["image"],
                prob=noise_prob,
                std=noise_std,
            )
        )
    
    # Contrast adjustment
    if contrast_prob > 0:
        transforms.append(
            RandAdjustContrastd(
                keys=["image"],
                prob=contrast_prob,
                gamma=contrast_gamma,
            )
        )
    
    # Intensity scaling
    if scale_intensity_prob > 0:
        transforms.append(
            RandScaleIntensityd(
                keys=["image"],
                prob=scale_intensity_prob,
                factors=scale_intensity_factors,
            )
        )
    
    # Intensity shifting
    if shift_intensity_prob > 0:
        transforms.append(
            RandShiftIntensityd(
                keys=["image"],
                prob=shift_intensity_prob,
                offsets=shift_intensity_offset,
            )
        )
    
    # Convert to tensor
    transforms.append(ToTensord(keys=["image"]))
    
    return Compose(transforms)


class TwoCropsTransform:
    """
    Wrapper that creates two augmented views from a single sample.
    This is the core of SimCLR's view generation.
    """
    
    def __init__(
        self,
        base_transform: Compose,
        patch_transform: Compose,
        aug_transform: Compose,
        num_crops_per_scan: int = 1,
    ):
        """
        Args:
            base_transform: Base preprocessing (load, normalize, crop foreground)
            patch_transform: Patch sampling (random crop + resize)
            aug_transform: Stochastic augmentation
            num_crops_per_scan: Number of crops per scan (K). If > 1, returns K pairs.
        """
        self.base_transform = base_transform
        self.patch_transform = patch_transform
        self.aug_transform = aug_transform
        self.num_crops_per_scan = num_crops_per_scan
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns dict with:
        - "id": identifier
        - "view1": tensor (K, C, D, H, W) or (C, D, H, W) if K=1
        - "view2": tensor (K, C, D, H, W) or (C, D, H, W) if K=1
        """
        # Apply base preprocessing
        sample = self.base_transform(sample)
        
        # Sample K patches
        view1_list = []
        view2_list = []
        
        for _ in range(self.num_crops_per_scan):
            # Sample a patch (this is the "instance")
            patch_sample = copy.deepcopy(sample)
            patch_sample = self.patch_transform(patch_sample)
            
            # Create two augmented views from the same patch
            view1_sample = copy.deepcopy(patch_sample)
            view2_sample = copy.deepcopy(patch_sample)
            
            view1 = self.aug_transform(view1_sample)["image"]
            view2 = self.aug_transform(view2_sample)["image"]
            
            view1_list.append(view1)
            view2_list.append(view2)
        
        # Stack if multiple crops
        if self.num_crops_per_scan > 1:
            view1 = torch.stack(view1_list, dim=0)  # (K, C, D, H, W)
            view2 = torch.stack(view2_list, dim=0)
        else:
            view1 = view1_list[0]  # (C, D, H, W)
            view2 = view2_list[0]
        
        return {
            "id": sample["id"],
            "view1": view1,
            "view2": view2,
        }
