"""
Dataset module for BrainIAC pretraining.
Loads images from JSON metadata and returns raw image tensors.
"""
import json
import copy
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    NormalizeIntensityd,
    EnsureTyped,
)


class PretrainDataset(Dataset):
    """
    Dataset for pretraining that loads images from JSON metadata.
    
    Each JSON record should have:
    - "id": unique identifier
    - "image": path to NIfTI file (or dict for multi-sequence)
    """
    
    def __init__(
        self,
        json_path: str,
        transform: Optional[Compose] = None,
    ):
        """
        Args:
            json_path: Path to JSON file with metadata
            transform: MONAI Compose transform for base preprocessing
        """
        self.json_path = json_path
        self.transform = transform
        
        # Load JSON metadata
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Validate metadata format
        if not isinstance(self.metadata, list):
            raise ValueError("JSON metadata must be a list of records")
        
        # Filter out invalid entries
        self.valid_indices = []
        for idx, record in enumerate(self.metadata):
            if self._is_valid_record(record):
                self.valid_indices.append(idx)
        
        print(f"Loaded {len(self.valid_indices)} valid records from {json_path}")
    
    def _is_valid_record(self, record: Dict) -> bool:
        """Check if a record is valid."""
        if not isinstance(record, dict):
            return False
        if "id" not in record:
            return False
        if "image" not in record:
            return False
        
        # Handle both single path and dict of paths
        if isinstance(record["image"], str):
            return Path(record["image"]).exists()
        elif isinstance(record["image"], dict):
            # Multi-sequence: check if at least one exists
            return any(Path(p).exists() for p in record["image"].values() if isinstance(p, str))
        return False
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict with:
        - "id": identifier
        - "image": preprocessed image tensor (C, D, H, W)
        - "meta": metadata dict
        """
        record_idx = self.valid_indices[idx]
        record = self.metadata[record_idx]
        
        # Handle single image or multi-sequence
        if isinstance(record["image"], str):
            image_path = record["image"]
        elif isinstance(record["image"], dict):
            # For multi-sequence, use first available sequence
            # In practice, you might want to sample randomly or use a specific sequence
            image_path = next(iter(record["image"].values()))
        else:
            raise ValueError(f"Invalid image format in record {record['id']}")
        
        # Create sample dict for MONAI transforms
        sample = {
            "id": record["id"],
            "image": image_path,
            "meta": copy.deepcopy(record)
        }
        
        # Apply base preprocessing transform
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
