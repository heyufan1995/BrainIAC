#!/usr/bin/env python3
"""
Utility script to create JSON metadata file from MR-RATE dataset structure.

This script scans a MR-RATE directory structure and creates a JSON metadata file
for BrainIAC pretraining.

Usage:
    python scripts/create_metadata_from_mrrate.py \
        --data_dir /path/to/mri \
        --output metadata.json \
        --sequences t1w t2w flair \
        --orientations axi
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
import re


def find_batch_directories(data_dir: Path) -> List[Path]:
    """
    Find all batch directories in MR-RATE structure.
    
    Args:
        data_dir: Root directory (e.g., mri/)
    
    Returns:
        List of batch directories
    """
    batches = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith('batch'):
            batches.append(item)
    return sorted(batches)


def find_subject_directories(batch_dir: Path) -> List[Path]:
    """
    Find all subject directories in a batch.
    
    Args:
        batch_dir: Batch directory (e.g., batch00)
    
    Returns:
        List of subject directories
    """
    subjects = []
    for item in batch_dir.iterdir():
        if item.is_dir() and item.name.isalnum():  # Subject IDs like "22B7CXEZ6T"
            subjects.append(item)
    return sorted(subjects)


def parse_image_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse MR-RATE image filename to extract sequence and orientation.
    
    Format: {subject_id}_{sequence}-raw-{orientation}[{-number}].nii.gz
    Examples:
        - 22B7CXEZ6T_t1w-raw-axi.nii.gz -> sequence: t1w, orientation: axi
        - 22B7CXEZ6T_t1w-raw-axi-2.nii.gz -> sequence: t1w, orientation: axi, version: 2
        - 22B7CXEZ6T_flair-raw-sag.nii.gz -> sequence: flair, orientation: sag
    
    Args:
        filename: Image filename
    
    Returns:
        Dict with 'sequence' and 'orientation' keys, or None if parsing fails
    """
    # Remove .nii.gz extension
    base = filename.replace('.nii.gz', '')
    
    # Pattern: {subject_id}_{sequence}-raw-{orientation}[-{number}]
    # We'll match the sequence and orientation parts
    pattern = r'^[^_]+_(.+?)-raw-(.+?)(?:-(\d+))?$'
    match = re.match(pattern, base)
    
    if match:
        sequence = match.group(1)  # e.g., 't1w', 'flair', 't2w'
        orientation = match.group(2)  # e.g., 'axi', 'cor', 'sag'
        version = match.group(3) if match.group(3) else None
        
        return {
            'sequence': sequence,
            'orientation': orientation,
            'version': version
        }
    
    return None


def find_images(
    subject_dir: Path,
    sequences: Optional[List[str]] = None,
    orientations: Optional[List[str]] = None,
    prefer_axi: bool = True,
) -> List[Dict[str, Path]]:
    """
    Find images for a subject.
    
    Args:
        subject_dir: Subject directory (e.g., 22B7CXEZ6T)
        sequences: List of sequence names to include (e.g., ['t1w', 't2w', 'flair'])
                   If None, includes all found sequences
        orientations: List of orientations to include (e.g., ['axi', 'cor', 'sag'])
                    If None, includes all found orientations
        prefer_axi: If True and multiple versions exist, prefer 'axi' orientation
    
    Returns:
        List of dicts with 'path', 'sequence', 'orientation', 'version' keys
    """
    img_dir = subject_dir / 'img'
    if not img_dir.exists():
        return []
    
    images = []
    for img_file in img_dir.glob('*.nii.gz'):
        parsed = parse_image_filename(img_file.name)
        if not parsed:
            continue
        
        sequence = parsed['sequence']
        orientation = parsed['orientation']
        version = parsed['version']
        
        # Filter by sequence
        if sequences is not None and sequence not in sequences:
            continue
        
        # Filter by orientation
        if orientations is not None and orientation not in orientations:
            continue
        
        images.append({
            'path': img_file,
            'sequence': sequence,
            'orientation': orientation,
            'version': version,
        })
    
    # If prefer_axi and multiple versions of same sequence exist, prefer axi
    if prefer_axi and orientations is None:
        # Group by sequence
        by_sequence = {}
        for img in images:
            seq = img['sequence']
            if seq not in by_sequence:
                by_sequence[seq] = []
            by_sequence[seq].append(img)
        
        # For each sequence, if multiple orientations exist, prefer axi
        filtered_images = []
        for seq, imgs in by_sequence.items():
            axi_imgs = [img for img in imgs if img['orientation'] == 'axi']
            if axi_imgs:
                # If multiple versions of axi, take the first one (or version=None)
                preferred = sorted(axi_imgs, key=lambda x: (x['version'] is None, x['version'] or ''))[0]
                filtered_images.append(preferred)
            else:
                # No axi, take first available
                filtered_images.append(imgs[0])
        
        return filtered_images
    
    return images


def create_metadata_from_mrrate(
    data_dir: str,
    output_path: str,
    sequences: Optional[List[str]] = None,
    orientations: Optional[List[str]] = None,
    prefer_axi: bool = True,
    max_subjects: Optional[int] = None,
) -> None:
    """
    Create JSON metadata file from MR-RATE dataset structure.
    
    Args:
        data_dir: Root directory containing batch directories (e.g., mri/)
        output_path: Path to save JSON file
        sequences: List of sequence names to include (e.g., ['t1w', 't2w', 'flair'])
                   If None, includes all found sequences
        orientations: List of orientations to include (e.g., ['axi', 'cor', 'sag'])
                     If None, includes all found orientations
        prefer_axi: If True, prefer 'axi' orientation when multiple exist
        max_subjects: Maximum number of subjects to process (None for all)
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    metadata = []
    
    # Find all batch directories
    batches = find_batch_directories(data_dir)
    if not batches:
        raise ValueError(f"No batch directories found in {data_dir}")
    
    print(f"Found {len(batches)} batch(es) to process")
    
    total_subjects = 0
    sequences_found: Set[str] = set()
    orientations_found: Set[str] = set()
    
    # Process each batch
    for batch_dir in sorted(batches):
        print(f"Processing batch: {batch_dir.name}")
        
        # Find all subjects
        subjects = find_subject_directories(batch_dir)
        print(f"  Found {len(subjects)} subjects")
        
        # Process each subject
        for subject_dir in subjects:
            if max_subjects and total_subjects >= max_subjects:
                print(f"  Reached max_subjects limit ({max_subjects})")
                break
            
            subject_id = subject_dir.name  # e.g., '22B7CXEZ6T'
            
            # Find images
            images = find_images(
                subject_dir,
                sequences=sequences,
                orientations=orientations,
                prefer_axi=prefer_axi,
            )
            
            if not images:
                continue
            
            # Create metadata entries (one per image)
            for img_info in images:
                sequence = img_info['sequence']
                orientation = img_info['orientation']
                version = img_info['version']
                
                sequences_found.add(sequence)
                orientations_found.add(orientation)
                
                # Create unique ID
                if version:
                    img_id = f"{batch_dir.name}_{subject_id}_{sequence}_{orientation}_{version}"
                else:
                    img_id = f"{batch_dir.name}_{subject_id}_{sequence}_{orientation}"
                
                metadata.append({
                    "id": img_id,
                    "image": str(img_info['path'].absolute()),
                    "meta": {
                        "batch": batch_dir.name,
                        "subject": subject_id,
                        "sequence": sequence,
                        "orientation": orientation,
                        "version": version,
                    }
                })
            
            total_subjects += 1
        
        if max_subjects and total_subjects >= max_subjects:
            break
    
    # Save JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCreated metadata file with {len(metadata)} entries: {output_path}")
    print(f"  Total subjects: {total_subjects}")
    print(f"  Total batches: {len(batches)}")
    print(f"  Sequences found: {', '.join(sorted(sequences_found))}")
    print(f"  Orientations found: {', '.join(sorted(orientations_found))}")


def main():
    parser = argparse.ArgumentParser(
        description="Create JSON metadata from MR-RATE dataset structure for BrainIAC pretraining"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing batch directories (e.g., /path/to/mri)",
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
        help="Sequence names to include (e.g., t1w t2w flair swi). If not specified, includes all found sequences",
    )
    parser.add_argument(
        "--orientations",
        type=str,
        nargs="+",
        default=None,
        help="Orientations to include (e.g., axi cor sag). If not specified, includes all found orientations",
    )
    parser.add_argument(
        "--no-prefer-axi",
        action="store_true",
        help="Don't prefer 'axi' orientation when multiple orientations exist",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Normalize sequence names to lowercase
    sequences = [s.lower() for s in args.sequences] if args.sequences else None
    orientations = [o.lower() for o in args.orientations] if args.orientations else None
    
    create_metadata_from_mrrate(
        data_dir=args.data_dir,
        output_path=args.output,
        sequences=sequences,
        orientations=orientations,
        prefer_axi=not args.no_prefer_axi,
        max_subjects=args.max_subjects,
    )


if __name__ == "__main__":
    main()
