"""
3D Vision Transformer backbone for BrainIAC.
Matches the architecture used in the paper: 96³ input, 16³ patches, ViT-Base.
"""
import torch
import torch.nn as nn
from monai.networks.nets import ViT


class ViT3D(nn.Module):
    """
    3D Vision Transformer backbone.
    
    Architecture:
    - Input: (B, 1, 96, 96, 96)
    - Patch size: (16, 16, 16) -> 6×6×6 = 216 patches
    - Hidden size: 768 (ViT-Base)
    - MLP dim: 3072
    - Layers: 12
    - Heads: 12
    - Output: CLS token embedding (B, 768)
    """
    
    def __init__(
        self,
        img_size: tuple = (96, 96, 96),
        patch_size: tuple = (16, 16, 16),
        in_channels: int = 1,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        save_attn: bool = False,
    ):
        """
        Args:
            img_size: Input image size (D, H, W)
            patch_size: Patch size (D, H, W)
            in_channels: Input channels
            hidden_size: Hidden dimension
            mlp_dim: MLP dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            save_attn: Whether to save attention maps
        """
        super().__init__()
        
        self.backbone = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            save_attn=save_attn,
        )
        
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            CLS token embedding (B, hidden_size)
        """
        # ViT returns tuple: (features, hidden_states, attentions)
        # features[0] is the output of all layers: (B, num_patches+1, hidden_size)
        # features[0][:, 0] is the CLS token: (B, hidden_size)
        features = self.backbone(x)
        cls_token = features[0][:, 0]  # Extract CLS token
        return cls_token
