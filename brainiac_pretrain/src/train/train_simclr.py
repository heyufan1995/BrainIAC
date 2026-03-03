"""
PyTorch Lightning training module for SimCLR pretraining.
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from typing import Optional, Dict, Any

# Use absolute imports since src is added to sys.path
from models.simclr import SimCLRModel
from losses.nt_xent import NTXentLoss
from utils.checkpoint import save_checkpoint


class SimCLRLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for SimCLR pretraining.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Build model
        model_config = config.get("model", {})
        self.model = SimCLRModel(
            img_size=tuple(model_config.get("img_size", [96, 96, 96])),
            patch_size=tuple(model_config.get("patch_size", [16, 16, 16])),
            in_channels=model_config.get("in_channels", 1),
            hidden_size=model_config.get("hidden_size", 768),
            mlp_dim=model_config.get("mlp_dim", 3072),
            num_layers=model_config.get("num_layers", 12),
            num_heads=model_config.get("num_heads", 12),
            proj_input_dim=model_config.get("proj_input_dim", 768),
            proj_hidden_dim=model_config.get("proj_hidden_dim", 512),
            proj_output_dim=model_config.get("proj_output_dim", 128),
        )
        
        # Loss function
        loss_config = config.get("loss", {})
        self.criterion = NTXentLoss(
            temperature=loss_config.get("temperature", 0.07)
        )
        
        # Training state
        self.num_crops_per_scan = config.get("data", {}).get("num_crops_per_scan", 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Dict with "view1" and "view2" keys
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        view1 = batch["view1"]  # (B, C, D, H, W) or (B, K, C, D, H, W) if multi-crop
        view2 = batch["view2"]
        
        # Handle multi-crop case: flatten K dimension into batch
        if self.num_crops_per_scan > 1:
            B, K, C, D, H, W = view1.shape
            view1 = view1.view(B * K, C, D, H, W)
            view2 = view2.view(B * K, C, D, H, W)
        
        # Forward pass through model
        z1 = self.model(view1)  # (B, proj_dim) or (B*K, proj_dim)
        z2 = self.model(view2)
        
        # Compute contrastive loss
        loss = self.criterion(z1, z2)
        
        # Logging
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # For DDP
        )
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        train_config = self.config.get("train", {})
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=train_config.get("lr", 1e-4),
            weight_decay=train_config.get("weight_decay", 1e-4),
            betas=tuple(train_config.get("betas", [0.9, 0.999])),
        )
        
        # Learning rate scheduler
        scheduler_config = train_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine")
        
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_config.get("max_epochs", 100),
                eta_min=scheduler_config.get("eta_min", 0.0),
            )
        elif scheduler_type == "cosine_warmup":
            # Cosine with warmup
            warmup_epochs = scheduler_config.get("warmup_epochs", 10)
            max_epochs = train_config.get("max_epochs", 100)
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get("step_size", 30),
                gamma=scheduler_config.get("gamma", 0.1),
            )
        else:
            scheduler = None
        
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer


def train_simclr(config_path: str, resume_from: Optional[str] = None):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML configuration file
        resume_from: Path to checkpoint to resume from (optional)
    """
    import yaml
    from data.datamodule import PretrainDataModule
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data module
    data_config = config.get("data", {})
    datamodule = PretrainDataModule(
        json_path=data_config["json_path"],
        batch_size=data_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        # Base preprocessing
        normalize_nonzero=data_config.get("normalize_nonzero", True),
        normalize_channel_wise=data_config.get("normalize_channel_wise", True),
        orientation=data_config.get("orientation", "RAS"),
        spacing=data_config.get("spacing"),
        # Patch sampling
        patch_roi=tuple(data_config.get("patch_roi", [96, 96, 96])),
        resize_to=tuple(data_config.get("resize_to", [96, 96, 96])) if data_config.get("resize_to") else None,
        num_crops_per_scan=data_config.get("num_crops_per_scan", 1),
        # Augmentation
        flip_prob=data_config.get("flip_prob", 0.5),
        flip_axes=data_config.get("flip_axes"),
        affine_prob=data_config.get("affine_prob", 0.5),
        rotate_range=tuple(data_config.get("rotate_range", [0.1, 0.1, 0.1])),
        translate_range=tuple(data_config.get("translate_range", [5, 5, 5])),
        scale_range=tuple(data_config.get("scale_range", [0.1, 0.1, 0.1])),
        noise_prob=data_config.get("noise_prob", 0.2),
        noise_std=data_config.get("noise_std", 0.05),
        blur_prob=data_config.get("blur_prob", 0.2),
        blur_sigma=tuple(data_config.get("blur_sigma", [0.5, 1.0])),
        contrast_prob=data_config.get("contrast_prob", 0.2),
        contrast_gamma=tuple(data_config.get("contrast_gamma", [0.7, 1.3])),
        scale_intensity_prob=data_config.get("scale_intensity_prob", 0.1),
        scale_intensity_factors=tuple(data_config.get("scale_intensity_factors", [0.8, 1.2])),
        shift_intensity_prob=data_config.get("shift_intensity_prob", 0.1),
        shift_intensity_offset=tuple(data_config.get("shift_intensity_offset", [-0.1, 0.1])),
    )
    
    # Create model
    model = SimCLRLightningModule(config)
    
    # Setup logger
    logger_config = config.get("logger", {})
    logger_type = logger_config.get("type", "tensorboard")
    
    if logger_type == "wandb":
        logger = WandbLogger(
            project=logger_config.get("project_name", "brainiac_pretrain"),
            name=logger_config.get("run_name", "simclr"),
            save_dir=logger_config.get("save_dir", "./logs"),
        )
    else:
        logger = TensorBoardLogger(
            save_dir=logger_config.get("save_dir", "./logs"),
            name=logger_config.get("run_name", "simclr"),
        )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger_config.get("checkpoint_dir", "./checkpoints"),
        filename=logger_config.get("checkpoint_filename", "brainiac-{epoch:02d}-{train_loss:.4f}"),
        monitor="train_loss",
        mode="min",
        save_top_k=logger_config.get("save_top_k", 3),
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Setup trainer
    train_config = config.get("train", {})
    trainer_config = config.get("trainer", {})
    
    trainer = pl.Trainer(
        max_epochs=train_config.get("max_epochs", 100),
        accelerator=trainer_config.get("accelerator", "gpu"),
        devices=trainer_config.get("devices", "auto"),
        strategy=trainer_config.get("strategy", "ddp"),
        precision=train_config.get("precision", "16-mixed"),
        gradient_clip_val=train_config.get("gradient_clip_val", 1.0),
        logger=logger,
        callbacks=callbacks,
        sync_batchnorm=trainer_config.get("sync_batchnorm", True),
        log_every_n_steps=trainer_config.get("log_every_n_steps", 50),
        val_check_interval=trainer_config.get("val_check_interval", None),
        enable_progress_bar=trainer_config.get("enable_progress_bar", True),
        enable_model_summary=trainer_config.get("enable_model_summary", True),
    )
    
    # Train
    trainer.fit(model, datamodule, ckpt_path=resume_from)
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_callback.best_model_path
    print(f"Training complete! Best checkpoint: {final_checkpoint_path}")
