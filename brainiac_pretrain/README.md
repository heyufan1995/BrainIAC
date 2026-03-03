# BrainIAC Pretraining

This repository contains the training code for **BrainIAC** (Brain Imaging Adaptive Core), a foundation model for brain MRI analysis based on self-supervised learning (SimCLR).

## Overview

BrainIAC is a 3D Vision Transformer (ViT) pretrained using contrastive learning on large-scale unlabeled brain MRI data. The model learns generalizable representations that can be fine-tuned for various downstream tasks such as:

- Brain age prediction
- Disease classification
- Survival prediction
- Tumor segmentation
- And more...

## Architecture

- **Backbone**: 3D Vision Transformer (ViT-Base)
  - Input size: 96×96×96 voxels
  - Patch size: 16×16×16 voxels (216 patches)
  - Hidden size: 768
  - Layers: 12
  - Attention heads: 12

- **Projection Head**: 2-layer MLP
  - Input: 768 (backbone output)
  - Hidden: 512
  - Output: 128 (normalized)

- **Training Objective**: NT-Xent (InfoNCE) contrastive loss

## Installation

### Requirements

```bash
pip install torch torchvision
pip install pytorch-lightning
pip install monai
pip install nibabel
pip install pyyaml
pip install wandb  # Optional, for Weights & Biases logging
```

Or install from the main repository's `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

## Data Preparation

### JSON Metadata Format

Create a JSON file listing all training images. Each record should have:

```json
[
  {
    "id": "subj001",
    "image": "/path/to/subj001.nii.gz"
  },
  {
    "id": "subj002",
    "image": "/path/to/subj002.nii.gz"
  }
]
```

For multi-sequence data (optional):

```json
[
  {
    "id": "subj001",
    "image": {
      "t1": "/path/to/subj001_t1.nii.gz",
      "flair": "/path/to/subj001_flair.nii.gz"
    }
  }
]
```

### Create Sample JSON

You can use the utility script to create a sample JSON:

```python
from src.utils.io import create_sample_json

create_sample_json(
    output_path="./data/train_metadata.json",
    image_dir="/path/to/nifti/files",
    num_samples=100  # Or None for all files
)
```

## Training

### Configuration

Edit `configs/pretrain_simclr.yaml` to set your data paths and training parameters.

Key configuration sections:

- **data**: Data paths, batch size, augmentation parameters
- **model**: Architecture parameters
- **loss**: Temperature parameter
- **train**: Learning rate, epochs, scheduler
- **trainer**: DDP settings, precision
- **logger**: Logging configuration

### Single GPU Training

```bash
python train.py --config configs/pretrain_simclr.yaml
```

### Multi-GPU Training (DDP)

Using `torchrun` (recommended):

```bash
torchrun --nproc_per_node=4 train.py --config configs/pretrain_simclr.yaml
```

Or use the provided script:

```bash
bash scripts/run_ddp.sh
```

### Resume Training

```bash
python train.py \
    --config configs/pretrain_simclr.yaml \
    --resume_from checkpoints/brainiac-epoch=50.ckpt
```

## Training Details

### Data Pipeline

1. **Base Preprocessing**:
   - Load NIfTI image
   - Ensure channel first
   - Optional: Reorient to RAS, resample to target spacing
   - Crop to foreground (brain region)
   - Normalize intensity (nonzero voxels)

2. **Patch Sampling**:
   - Randomly crop a patch from the foreground region
   - Resize to 96×96×96 (if needed)

3. **View Generation**:
   - Create two augmented views from the same patch
   - Augmentations include: rotation, translation, scaling, flipping, noise, blur, contrast adjustment

4. **Multi-Crop (Optional)**:
   - Sample K patches per scan
   - Increases negative samples and training efficiency

### Loss Function

NT-Xent (Normalized Temperature-scaled Cross Entropy) loss:

- Positive pairs: Two augmented views of the same patch
- Negative pairs: All other patches in the batch
- Temperature scaling: τ = 0.07 (default)

### Training Tips

1. **Batch Size**: Use as large a batch size as possible (important for contrastive learning)
2. **Learning Rate**: Start with 1e-4, adjust based on batch size
3. **Warmup**: Use cosine annealing with warmup (10-20 epochs)
4. **Mixed Precision**: Use `16-mixed` for faster training
5. **Multi-Crop**: Set `num_crops_per_scan > 1` to increase negatives

## Output

Training produces:

- **Checkpoints**: Saved in `checkpoints/` directory
  - Best model: `brainiac-epoch=XX-train_loss=X.XXXX.ckpt`
  - Last model: `last.ckpt`
  - Backbone-only checkpoints (for fine-tuning): `backbone_epoch_XX.pt`

- **Logs**: TensorBoard or Weights & Biases logs in `logs/`

## Using Pretrained Weights

After training, extract the backbone weights for downstream fine-tuning:

```python
import torch
from src.models.simclr import SimCLRModel

# Load checkpoint
checkpoint = torch.load("checkpoints/brainiac-epoch=50.ckpt", map_location="cpu")
state_dict = checkpoint["state_dict"]

# Extract backbone weights
backbone_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("model.backbone."):
        new_key = key.replace("model.backbone.", "backbone.")
        backbone_state_dict[new_key] = value

# Save backbone-only checkpoint
torch.save(backbone_state_dict, "brainiac_backbone.pt")
```

Then use in downstream tasks (see main repository's fine-tuning scripts).

## Repository Structure

```
brainiac_pretrain/
├── configs/
│   └── pretrain_simclr.yaml      # Training configuration
├── src/
│   ├── data/
│   │   ├── dataset.py            # Dataset class
│   │   ├── transforms.py         # Transform pipelines
│   │   ├── sampler.py            # Distributed sampler
│   │   └── datamodule.py         # PyTorch Lightning DataModule
│   ├── models/
│   │   ├── vit3d.py              # 3D ViT backbone
│   │   ├── heads.py              # Projection head
│   │   └── simclr.py              # SimCLR model wrapper
│   ├── losses/
│   │   └── nt_xent.py            # NT-Xent loss
│   ├── train/
│   │   ├── train_simclr.py       # Training module
│   │   ├── ddp_utils.py          # DDP utilities
│   │   └── callbacks.py          # Custom callbacks
│   └── utils/
│       ├── io.py                 # I/O utilities
│       ├── checkpoint.py         # Checkpoint utilities
│       ├── logging.py            # Logging setup
│       └── misc.py                # Miscellaneous utilities
├── scripts/
│   └── run_ddp.sh                # DDP training script
├── train.py                      # Main training script
└── README.md                     # This file
```

## Citation

If you use this code, please cite the BrainIAC paper:

```bibtex
@article{tak2026generalizable,
  title={A generalizable foundation model for analysis of human brain MRI},
  author={Tak, Divyanshu and others},
  journal={Nature Neuroscience},
  year={2026}
}
```

## License

See the main repository's LICENSE file.

## Contact

For questions or issues, please open an issue in the main repository.
