#!/usr/bin/env python3
"""
Script to analyze OpenMind dataset and extract all modalities from filenames.

This script scans all .nii.gz files in the OpenMind dataset and extracts
the modality information from filenames.

Usage:
    python scripts/analyze_openmind_modalities.py \
        --data_dir /path/to/OpenMind \
        --output modalities_report.txt
"""
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import json


def extract_modality_from_filename(filename: str) -> str:
    """
    Extract modality from OpenMind filename.
    
    Format: sub-XX_acq-XXX_run-XX_MODALITY.nii.gz
    Examples:
        - sub-11_acq-sagittal_run-01_T1w.nii.gz -> T1w
        - sub-11_acq-axial_run-01_T2w.nii.gz -> T2w
        - sub-11_acq-coronal_run-01_FLAIR.nii.gz -> FLAIR
    
    Args:
        filename: Full filename (e.g., "sub-11_acq-sagittal_run-01_T1w.nii.gz")
    
    Returns:
        Modality string (e.g., "T1w") or "unknown" if parsing fails
    """
    # Remove .nii.gz extension
    base = filename.replace('.nii.gz', '')
    
    # Split by underscore
    parts = base.split('_')
    
    # The modality should be the last part
    if len(parts) > 0:
        modality = parts[-1]
        # Clean up any remaining extensions
        modality = modality.strip()
        return modality if modality else "unknown"
    
    return "unknown"


def find_all_nifti_files(data_dir: Path) -> list:
    """
    Recursively find all .nii.gz files in the directory.
    
    Args:
        data_dir: Root directory to search
    
    Returns:
        List of Path objects for all .nii.gz files
    """
    nifti_files = list(data_dir.rglob('*.nii.gz'))
    return sorted(nifti_files)


def analyze_modalities(data_dir: str, output_path: str = None) -> dict:
    """
    Analyze all .nii.gz files and extract modality information.
    
    Args:
        data_dir: Root directory containing OpenMind dataset
        output_path: Optional path to save report (if None, prints to stdout)
    
    Returns:
        Dictionary with analysis results
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    print(f"Scanning directory: {data_dir}")
    print("Finding all .nii.gz files...")
    
    # Find all .nii.gz files
    nifti_files = find_all_nifti_files(data_dir)
    
    if not nifti_files:
        print(f"Warning: No .nii.gz files found in {data_dir}")
        return {}
    
    print(f"Found {len(nifti_files)} .nii.gz files")
    print("Analyzing modalities...")
    
    # Extract modalities
    modality_counter = Counter()
    modality_files = defaultdict(list)
    subject_modalities = defaultdict(set)
    unknown_files = []
    
    for nifti_file in nifti_files:
        filename = nifti_file.name
        modality = extract_modality_from_filename(filename)
        
        modality_counter[modality] += 1
        modality_files[modality].append(str(nifti_file))
        
        # Extract subject ID (sub-XX)
        subject_id = None
        if filename.startswith('sub-'):
            parts = filename.split('_')
            if len(parts) > 0:
                subject_id = parts[0]  # e.g., "sub-11"
        
        if subject_id:
            subject_modalities[subject_id].add(modality)
        
        if modality == "unknown":
            unknown_files.append(str(nifti_file))
    
    # Compile results
    results = {
        "total_files": len(nifti_files),
        "total_subjects": len(subject_modalities),
        "modalities": dict(modality_counter),
        "modality_counts": dict(modality_counter),
        "unknown_files": unknown_files,
        "subject_modality_summary": {
            subject: sorted(list(modalities))
            for subject, modalities in subject_modalities.items()
        }
    }
    
    # Print results
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("OpenMind Dataset Modality Analysis")
    output_lines.append("=" * 80)
    output_lines.append(f"\nTotal .nii.gz files found: {len(nifti_files)}")
    output_lines.append(f"Total unique subjects: {len(subject_modalities)}")
    output_lines.append(f"Total unique modalities: {len(modality_counter)}")
    
    output_lines.append("\n" + "-" * 80)
    output_lines.append("Modality Counts:")
    output_lines.append("-" * 80)
    for modality, count in modality_counter.most_common():
        percentage = (count / len(nifti_files)) * 100
        output_lines.append(f"  {modality:20s}: {count:6d} files ({percentage:5.1f}%)")
    
    if unknown_files:
        output_lines.append("\n" + "-" * 80)
        output_lines.append(f"Unknown Modalities ({len(unknown_files)} files):")
        output_lines.append("-" * 80)
        for filepath in unknown_files[:20]:  # Show first 20
            output_lines.append(f"  {filepath}")
        if len(unknown_files) > 20:
            output_lines.append(f"  ... and {len(unknown_files) - 20} more")
    
    output_lines.append("\n" + "-" * 80)
    output_lines.append("Modality Distribution by Subject:")
    output_lines.append("-" * 80)
    
    # Count how many subjects have each modality combination
    modality_combinations = Counter()
    for subject, modalities in subject_modalities.items():
        combo = tuple(sorted(modalities))
        modality_combinations[combo] += 1
    
    output_lines.append("\nMost common modality combinations:")
    for combo, count in modality_combinations.most_common(10):
        modalities_str = ", ".join(combo)
        output_lines.append(f"  [{modalities_str}]: {count} subjects")
    
    output_lines.append("\n" + "-" * 80)
    output_lines.append("Sample Files by Modality:")
    output_lines.append("-" * 80)
    for modality in sorted(modality_counter.keys()):
        if modality != "unknown":
            sample_files = modality_files[modality][:3]  # Show first 3
            output_lines.append(f"\n{modality}:")
            for filepath in sample_files:
                output_lines.append(f"  {filepath}")
    
    output_lines.append("\n" + "=" * 80)
    
    # Print or save
    report_text = "\n".join(output_lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_path}")
    else:
        print("\n" + report_text)
    
    # Also save JSON summary
    if output_path:
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON summary saved to: {json_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze OpenMind dataset to extract modality information from filenames"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing OpenMind dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for report (optional, prints to stdout if not specified)",
    )
    
    args = parser.parse_args()
    
    analyze_modalities(
        data_dir=args.data_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
