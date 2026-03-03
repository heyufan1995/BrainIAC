"""
SimCLR model wrapper combining backbone and projection head.
"""
import torch
import torch.nn as nn
from typing import Optional

from .vit3d import ViT3D
from .heads import ProjectionHead


class SimCLRModel(nn.Module):
    """
    SimCLR model: ViT backbone + projection head.
    
    Forward pass:
    - view -> backbone -> projection -> normalized z
    """
    
    def __init__(
        self,
        backbone: Optional[ViT3D] = None,
        projection_head: Optional[ProjectionHead] = None,
        # Backbone args (used if backbone is None)
        img_size: tuple = (96, 96, 96),
        patch_size: tuple = (16, 16, 16),
        in_channels: int = 1,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        # Projection head args (used if projection_head is None)
        proj_input_dim: int = 768,
        proj_hidden_dim: int = 512,
        proj_output_dim: int = 128,
    ):
        """
        Args:
            backbone: ViT3D backbone (if None, creates new one)
            projection_head: ProjectionHead (if None, creates new one)
            Other args: Used to create backbone/projection if not provided
        """
        super().__init__()
        
        if backbone is None:
            self.backbone = ViT3D(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        else:
            self.backbone = backbone
        
        if projection_head is None:
            self.projection_head = ProjectionHead(
                input_dim=proj_input_dim,
                hidden_dim=proj_hidden_dim,
                output_dim=proj_output_dim,
            )
        else:
            self.projection_head = projection_head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone and projection head.
        
        Args:
            x: Input tensor (B, C, D, H, W) or (B*K, C, D, H, W) if multi-crop
        
        Returns:
            Projected embeddings (B, proj_dim) or (B*K, proj_dim)
        """
        # Get features from backbone
        h = self.backbone(x)  # (B, hidden_size)
        
        # Project to lower dimension
        z = self.projection_head(h)  # (B, proj_dim)
        
        return z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode without projection (for feature extraction).
        
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            Backbone features (B, hidden_size)
        """
        return self.backbone(x)
