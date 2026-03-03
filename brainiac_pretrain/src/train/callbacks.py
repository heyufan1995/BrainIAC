"""
Custom callbacks for training.
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Optional


class SaveBackboneCallback(Callback):
    """
    Callback to save backbone weights separately (without projection head).
    Useful for downstream fine-tuning.
    """
    
    def __init__(self, save_dir: str = "./checkpoints", save_freq: int = 10):
        """
        Args:
            save_dir: Directory to save checkpoints
            save_freq: Save every N epochs
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = save_freq
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save backbone at end of epoch."""
        if (trainer.current_epoch + 1) % self.save_freq == 0:
            import torch
            from pathlib import Path
            
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            
            # Extract backbone state dict
            backbone_state_dict = {}
            for key, value in pl_module.model.backbone.state_dict().items():
                backbone_state_dict[f"backbone.{key}"] = value
            
            # Save
            checkpoint_path = Path(self.save_dir) / f"backbone_epoch_{trainer.current_epoch + 1}.pt"
            torch.save(backbone_state_dict, checkpoint_path)
            
            if pl_module.global_rank == 0:
                print(f"Saved backbone checkpoint: {checkpoint_path}")
