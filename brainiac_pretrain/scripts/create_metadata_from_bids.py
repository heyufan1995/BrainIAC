#!/usr/bin/env python3
"""
Utility script to create JSON metadata file from BIDS-structured dataset.

This script scans a BIDS directory structure and creates a JSON metadata file
for BrainIAC pretraining.

Usage:
    python scripts/create_metadata_from_bids.py \
        --bids_dir /path/to/OpenMind \
        --output metadata.json \
        --sequences T1w inplaneT2
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import os


def find_bids_subjects(bids_dir: Path) -> List[Path]:
    """
    Find all subject directories in BIDS structure.
    
    Args:
        bids_dir: Root BIDS directory (e.g., OpenMind/ds000001)
    
    Returns:
        List of subject directories
    """
    subjects = []
    for item in bids_dir.iterdir():
        if item.is_dir() and item.name.startswith('sub-'):
            subjects.append(item)
    return sorted(subjects)


def find_anat_images(subject_dir: Path, sequences: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Find anatomical images for a subject.
    
    Args:
        subject_dir: Subject directory (e.g., sub-01)
        sequences: List of sequence names to include (e.g., ['T1w', 'inplaneT2'])
                   If None, includes all found sequences
    
    Returns:
        Dict mapping sequence name to image path
    """
    anat_dir = subject_dir / 'anat'
    if not anat_dir.exists():
        return {}
    
    images = {}
    for item in anat_dir.iterdir():
        if item.is_file() and item.suffix == '.gz' and not item.name.endswith('mask.nii.gz'):
            # Extract sequence name from filename
            # Format: sub-XX_SequenceName.nii.gz
            parts = item.stem.replace('.nii', '').split('_')
            if len(parts) >= 2:
                sequence = parts[1]  # e.g., 'T1w' or 'inplaneT2'
                
                # Filter by requested sequences
                if sequences is None or sequence in sequences:
                    images[sequence] = item
    
    return images


def create_metadata_from_bids(
    bids_root: str,
    output_path: str,
    sequences: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    recursive: bool = True,
) -> None:
    """
    Create JSON metadata file from BIDS-structured dataset.
    
    Args:
        bids_root: Root directory containing BIDS datasets (e.g., OpenMind/)
        output_path: Path to save JSON file
        sequences: List of sequence names to include (e.g., ['T1w', 'inplaneT2'])
                   If None, includes all sequences found
        dataset_name: Specific dataset to process (e.g., 'ds000001'). If None, processes all
        recursive: Search recursively for BIDS datasets
    """
    bids_root = Path(bids_root)
    
    if not bids_root.exists():
        raise ValueError(f"BIDS root directory does not exist: {bids_root}")
    
    metadata = []
    
    # Find BIDS datasets
    if dataset_name:
        # Process specific dataset
        dataset_dirs = [bids_root / dataset_name]
    else:
        # Find all dataset directories
        if recursive:
            # Look for ds-* directories
            dataset_dirs = [d for d in bids_root.rglob('ds-*') if d.is_dir()]
            # Also check direct children
            dataset_dirs.extend([d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith('ds')])
        else:
            dataset_dirs = [d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith('ds')]
    
    if not dataset_dirs:
        raise ValueError(f"No BIDS datasets found in {bids_root}")
    
    print(f"Found {len(dataset_dirs)} dataset(s) to process")
    
    # Process each dataset
    for dataset_dir in sorted(dataset_dirs):
        print(f"Processing dataset: {dataset_dir.name}")
        
        # Find all subjects
        subjects = find_bids_subjects(dataset_dir)
        print(f"  Found {len(subjects)} subjects")
        
        # Process each subject
        for subject_dir in subjects:
            subject_id = subject_dir.name  # e.g., 'sub-01'
            
            # Find anatomical images
            images = find_anat_images(subject_dir, sequences=sequences)
            
            if not images:
                print(f"  Warning: No images found for {subject_id}")
                continue
            
            # Create metadata entries
            # Option 1: Create separate entry for each sequence (recommended for pretraining)
            # This gives more training samples and explicit sequence diversity
            for sequence_name, image_path in images.items():
                metadata.append({
                    "id": f"{dataset_dir.name}_{subject_id}_{sequence_name}",
                    "image": str(image_path.absolute()),
                    "meta": {
                        "dataset": dataset_dir.name,
                        "subject": subject_id,
                        "sequence": sequence_name,
                    }
                })
            
            # Option 2: Single entry with multiple sequences (alternative)
            # Uncomment below and comment above if you prefer this approach
            # The dataset loader will randomly sample one sequence per epoch
            # if len(images) == 1:
            #     # Single sequence: simple format
            #     sequence_name, image_path = next(iter(images.items()))
            #     metadata.append({
            #         "id": f"{dataset_dir.name}_{subject_id}_{sequence_name}",
            #         "image": str(image_path.absolute()),
            #         "meta": {
            #             "dataset": dataset_dir.name,
            #             "subject": subject_id,
            #             "sequence": sequence_name,
            #         }
            #     })
            # else:
            #     # Multiple sequences: use dict format
            #     image_dict = {seq: str(path.absolute()) for seq, path in images.items()}
            #     metadata.append({
            #         "id": f"{dataset_dir.name}_{subject_id}",
            #         "image": image_dict,
            #         "meta": {
            #             "dataset": dataset_dir.name,
            #             "subject": subject_id,
            #             "sequences": list(images.keys()),
            #         }
            #     })
    
    # Save JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCreated metadata file with {len(metadata)} entries: {output_path}")
    print(f"  Total subjects: {len(set(m['meta'].get('subject', '') for m in metadata))}")
    
    # Print sequence statistics
    if metadata:
        sequences_found = set()
        for m in metadata:
            if isinstance(m['image'], dict):
                sequences_found.update(m['image'].keys())
            else:
                seq = m['meta'].get('sequence', 'unknown')
                sequences_found.add(seq)
        print(f"  Sequences found: {', '.join(sorted(sequences_found))}")


def main():
    parser = argparse.ArgumentParser(
        description="Create JSON metadata from BIDS-structured dataset for BrainIAC pretraining"
    )
    parser.add_argument(
        "--bids_dir",
        type=str,
        required=True,
        help="Root directory containing BIDS datasets (e.g., /path/to/OpenMind)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Sequence names to include (e.g., T1w inplaneT2). If not specified, includes all found sequences",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to process (e.g., ds000001). If not specified, processes all datasets",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search recursively for BIDS datasets",
    )
    
    args = parser.parse_args()
    
    create_metadata_from_bids(
        bids_root=args.bids_dir,
        output_path=args.output,
        sequences=args.sequences,
        dataset_name=args.dataset,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
