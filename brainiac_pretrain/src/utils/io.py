"""
I/O utilities for loading and saving data.
"""
import json
from pathlib import Path
from typing import Dict, List, Any


def load_json(json_path: str) -> Any:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_json(data: Any, json_path: str, indent: int = 2):
    """Save data to JSON file."""
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=indent)


def create_sample_json(output_path: str, image_dir: str, num_samples: int = 10):
    """
    Create a sample JSON file for pretraining.
    
    Args:
        output_path: Path to save JSON file
        image_dir: Directory containing NIfTI files
        num_samples: Number of samples to include
    """
    image_dir = Path(image_dir)
    nifti_files = list(image_dir.glob("*.nii.gz"))
    
    if len(nifti_files) == 0:
        raise ValueError(f"No .nii.gz files found in {image_dir}")
    
    # Take first num_samples files
    nifti_files = nifti_files[:num_samples]
    
    # Create metadata
    metadata = []
    for nifti_file in nifti_files:
        metadata.append({
            "id": nifti_file.stem.replace(".nii", ""),
            "image": str(nifti_file.absolute()),
        })
    
    save_json(metadata, output_path)
    print(f"Created sample JSON with {len(metadata)} entries: {output_path}")
