# Using MR-RATE Dataset

This guide explains how to use MR-RATE formatted datasets with BrainIAC pretraining.

## MR-RATE Structure

MR-RATE datasets have the following structure:

```
mri/
├── batch00/
│   ├── 22B7CXEZ6T/
│   │   ├── img/
│   │   │   ├── 22B7CXEZ6T_t1w-raw-axi.nii.gz
│   │   │   ├── 22B7CXEZ6T_t1w-raw-cor.nii.gz
│   │   │   ├── 22B7CXEZ6T_t2w-raw-axi.nii.gz
│   │   │   └── ...
│   │   └── seg/
│   │       └── ... (masks, not used for pretraining)
│   ├── 22FM453NW2/
│   │   └── ...
│   └── ...
├── batch01/
│   └── ...
└── ...
```

## Step 1: Create Metadata JSON

Use the provided script to convert your MR-RATE structure to JSON metadata:

```bash
python brainiac_pretrain/scripts/create_metadata_from_mrrate.py \
    --data_dir /path/to/mri \
    --output metadata.json \
    --sequences t1w t2w flair \
    --orientations axi
```

### Options

- `--data_dir`: Root directory containing batch directories (e.g., `/path/to/mri`)
- `--output`: Path to save the JSON metadata file
- `--sequences`: (Optional) List of sequence names to include (e.g., `t1w t2w flair swi`). If not specified, includes all found sequences
- `--orientations`: (Optional) List of orientations to include (e.g., `axi cor sag`). If not specified, includes all found orientations
- `--no-prefer-axi`: Don't prefer 'axi' orientation when multiple orientations exist (default: prefers 'axi')
- `--max-subjects`: Maximum number of subjects to process (useful for testing)

### Examples

**Process all sequences and orientations:**
```bash
python brainiac_pretrain/scripts/create_metadata_from_mrrate.py \
    --data_dir /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/MR-RATE_20260227_unzip/mri \
    --output metadata.json
```

**Process only T1w and T2w in axial orientation (recommended for pretraining):**
```bash
python brainiac_pretrain/scripts/create_metadata_from_mrrate.py \
    --data_dir /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/datasets/MR-RATE/MR-RATE_20260227_unzip/mri \
    --output metadata.json \
    --sequences t1w t2w flair \
    --orientations axi
```

**Process specific sequences without orientation filtering:**
```bash
python brainiac_pretrain/scripts/create_metadata_from_mrrate.py \
    --data_dir /path/to/mri \
    --output metadata.json \
    --sequences t1w t2w flair swi \
    --no-prefer-axi
```

**Test with limited subjects:**
```bash
python brainiac_pretrain/scripts/create_metadata_from_mrrate.py \
    --data_dir /path/to/mri \
    --output metadata_test.json \
    --sequences t1w t2w \
    --orientations axi \
    --max-subjects 10
```

## Step 2: Generated JSON Format

The script generates a JSON file with entries like:

```json
[
  {
    "id": "batch00_22B7CXEZ6T_t1w_axi",
    "image": "/path/to/mri/batch00/22B7CXEZ6T/img/22B7CXEZ6T_t1w-raw-axi.nii.gz",
    "meta": {
      "batch": "batch00",
      "subject": "22B7CXEZ6T",
      "sequence": "t1w",
      "orientation": "axi",
      "version": null
    }
  },
  {
    "id": "batch00_22B7CXEZ6T_t2w_axi",
    "image": "/path/to/mri/batch00/22B7CXEZ6T/img/22B7CXEZ6T_t2w-raw-axi.nii.gz",
    "meta": {
      "batch": "batch00",
      "subject": "22B7CXEZ6T",
      "sequence": "t2w",
      "orientation": "axi",
      "version": null
    }
  }
]
```

## Step 3: Training with MR-RATE Data

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

## Filename Parsing

The script automatically parses MR-RATE filenames to extract:
- **Sequence**: t1w, t2w, flair, swi, etc.
- **Orientation**: axi, cor, sag, obl, etc.
- **Version**: Some sequences have multiple versions (e.g., `t1w-raw-axi-2.nii.gz`)

The filename format is: `{subject_id}_{sequence}-raw-{orientation}[-{version}].nii.gz`

## Recommendations

1. **For pretraining**: Use `--sequences t1w t2w flair --orientations axi` to get consistent axial slices
2. **Multiple orientations**: If you want to use multiple orientations, omit `--orientations` but use `--no-prefer-axi` to include all
3. **Sequence diversity**: Including multiple sequences (t1w, t2w, flair) helps the model learn sequence-invariant features
4. **Large datasets**: For very large datasets, consider processing in batches or using `--max-subjects` for testing

## Troubleshooting

**No images found:**
- Check that your directory structure matches: `mri/batchXX/subject_id/img/`
- Verify that `.nii.gz` files exist in the `img/` folder
- Ensure filenames follow the pattern: `{subject_id}_{sequence}-raw-{orientation}.nii.gz`

**Wrong sequences/orientations:**
- Use `--sequences` and `--orientations` to explicitly filter
- Check that sequence/orientation names in filenames match (case-sensitive, lowercase)

**Too many images:**
- Use `--orientations axi` to limit to axial slices only
- Use `--sequences t1w t2w` to limit to specific sequences
- Use `--max-subjects` to test with a subset
