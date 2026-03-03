# Utility Scripts

This directory contains utility scripts for working with BrainIAC pretraining datasets.

## Dataset Analysis

### `analyze_openmind_modalities.py`

Analyzes OpenMind dataset to extract and report all modalities found in filenames.

**Usage:**
```bash
python scripts/analyze_openmind_modalities.py \
    --data_dir /path/to/OpenMind \
    --output modalities_report.txt
```

**Output:**
- Text report with modality counts and statistics
- JSON summary file with detailed information
- Lists all unique modalities found
- Shows modality distribution across subjects

**Example:**
```bash
python scripts/analyze_openmind_modalities.py \
    --data_dir /path/to/OpenMind \
    --output openmind_modalities.txt
```

This will create:
- `openmind_modalities.txt` - Human-readable report
- `openmind_modalities.json` - Machine-readable summary

## Dataset Conversion

### `create_metadata_from_bids.py`

Converts BIDS-format datasets to JSON metadata for pretraining.

See [BIDS_DATASET.md](../docs/BIDS_DATASET.md) for details.

### `create_metadata_from_mrrate.py`

Converts MR-RATE-format datasets to JSON metadata for pretraining.

See [MR_RATE_DATASET.md](../docs/MR_RATE_DATASET.md) for details.
