#!/usr/bin/env python3
"""
Utility script to create JSON metadata file for pretraining.

Usage:
    python scripts/create_metadata.py \
        --image_dir /path/to/nifti/files \
        --output metadata.json \
        --pattern "*.nii.gz"
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict


def create_metadata(
    image_dir: str,
    output_path: str,
    pattern: str = "*.nii.gz",
    recursive: bool = True,
) -> None:
    """
    Create JSON metadata file from directory of NIfTI files.
    
    Args:
        image_dir: Directory containing NIfTI files
        output_path: Path to save JSON file
        pattern: Glob pattern for files (default: "*.nii.gz")
        recursive: Search recursively in subdirectories
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    # Find all matching files
    if recursive:
        nifti_files = list(image_dir.rglob(pattern))
    else:
        nifti_files = list(image_dir.glob(pattern))
    
    if len(nifti_files) == 0:
        raise ValueError(f"No files matching pattern '{pattern}' found in {image_dir}")
    
    # Create metadata
    metadata = []
    for nifti_file in sorted(nifti_files):
        # Use stem as ID (remove .nii.gz extension)
        file_id = nifti_file.stem.replace(".nii", "")
        
        metadata.append({
            "id": file_id,
            "image": str(nifti_file.absolute()),
        })
    
    # Save JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata file with {len(metadata)} entries: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create JSON metadata for pretraining")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing NIfTI files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.nii.gz",
        help="File pattern to match (default: '*.nii.gz')",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search recursively in subdirectories",
    )
    
    args = parser.parse_args()
    
    create_metadata(
        image_dir=args.image_dir,
        output_path=args.output,
        pattern=args.pattern,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
