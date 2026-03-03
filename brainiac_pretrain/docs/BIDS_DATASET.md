# Using BIDS-Formatted Datasets

This guide explains how to use BIDS (Brain Imaging Data Structure) formatted datasets with BrainIAC pretraining.

## BIDS Structure

BIDS datasets typically have the following structure:

```
OpenMind/
└── ds000001/
    ├── sub-01/
    │   └── anat/
    │       ├── sub-01_T1w.nii.gz
    │       └── sub-01_inplaneT2.nii.gz
    ├── sub-02/
    │   └── anat/
    │       ├── sub-02_T1w.nii.gz
    │       └── sub-02_inplaneT2.nii.gz
    └── ...
```

## Step 1: Create Metadata JSON

Use the provided script to convert your BIDS structure to JSON metadata:

```bash
python brainiac_pretrain/scripts/create_metadata_from_bids.py \
    --bids_dir /path/to/OpenMind \
    --output metadata.json \
    --sequences T1w inplaneT2
```

### Options

- `--bids_dir`: Root directory containing BIDS datasets (e.g., `/path/to/OpenMind`)
- `--output`: Path to save the JSON metadata file
- `--sequences`: (Optional) List of sequence names to include (e.g., `T1w inplaneT2`). If not specified, includes all found sequences
- `--dataset`: (Optional) Specific dataset to process (e.g., `ds000001`). If not specified, processes all datasets
- `--no-recursive`: Don't search recursively for BIDS datasets

### Examples

**Process all sequences from all datasets:**
```bash
python brainiac_pretrain/scripts/create_metadata_from_bids.py \
    --bids_dir /path/to/OpenMind \
    --output metadata.json
```

**Process only T1w sequences from a specific dataset:**
```bash
python brainiac_pretrain/scripts/create_metadata_from_bids.py \
    --bids_dir /path/to/OpenMind \
    --output metadata_t1w.json \
    --sequences T1w \
    --dataset ds000001
```

**Process multiple sequences:**
```bash
python brainiac_pretrain/scripts/create_metadata_from_bids.py \
    --bids_dir /path/to/OpenMind \
    --output metadata.json \
    --sequences T1w inplaneT2 T2w FLAIR
```

## Step 2: Generated JSON Format

The script generates a JSON file with two possible formats:

### Single Sequence Format
When a subject has only one sequence:
```json
[
  {
    "id": "ds000001_sub-01_T1w",
    "image": "/path/to/sub-01_T1w.nii.gz",
    "meta": {
      "dataset": "ds000001",
      "subject": "sub-01",
      "sequence": "T1w"
    }
  }
]
```

### Multi-Sequence Format
When a subject has multiple sequences:
```json
[
  {
    "id": "ds000001_sub-01",
    "image": {
      "T1w": "/path/to/sub-01_T1w.nii.gz",
      "inplaneT2": "/path/to/sub-01_inplaneT2.nii.gz"
    },
    "meta": {
      "dataset": "ds000001",
      "subject": "sub-01",
      "sequences": ["T1w", "inplaneT2"]
    }
  }
]
```

## Step 3: Training with BIDS Data

The dataset loader automatically handles both formats:

- **Single sequence entries**: Uses the specified image
- **Multi-sequence entries**: Randomly samples one sequence per epoch for diversity

This means if a subject has both T1w and T2 images, the model will see different sequences across epochs, which helps with generalization.

### Update Configuration

Edit `brainiac_pretrain/configs/pretrain_simclr.yaml`:

```yaml
data:
  json_path: "./metadata.json"  # Path to your generated JSON
  batch_size: 32
  # ... other settings
```

### Start Training

```bash
python brainiac_pretrain/train.py --config brainiac_pretrain/configs/pretrain_simclr.yaml
```

## Handling Multiple Sequences

The dataset loader handles multi-sequence subjects intelligently:

1. **During training**: For each epoch, it randomly selects one sequence per subject. This means:
   - Subject with T1w and T2: sometimes sees T1w, sometimes T2
   - Increases data diversity without duplicating subjects
   - Helps model learn sequence-invariant features

2. **If you want separate entries**: The BIDS script can be modified to create separate entries for each sequence (see script comments).

## Tips

1. **Sequence Selection**: If you have many sequences, you can filter to the most common ones (e.g., T1w, T2w) for faster training.

2. **Multiple Datasets**: The script automatically processes all `ds-*` directories found in the BIDS root.

3. **Validation**: The script validates that image files exist before adding them to metadata.

4. **Statistics**: The script prints summary statistics including:
   - Number of entries created
   - Number of unique subjects
   - Sequences found

## Troubleshooting

**No images found:**
- Check that your BIDS structure follows the expected format
- Verify that `.nii.gz` files exist (not just `.nii`)
- Ensure sequence names match (case-sensitive)

**Missing sequences:**
- Use `--sequences` to explicitly list sequences to include
- Check that sequence names in filenames match (e.g., `T1w` vs `t1w`)

**Multiple datasets:**
- Use `--dataset` to process one dataset at a time
- Or ensure all datasets follow the same structure
