"""
Projection heads for contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Standard SimCLR projection head: 2-layer MLP with BN + ReLU.
    
    Architecture:
    - Input: (B, input_dim)
    - Linear(input_dim, hidden_dim) -> BN -> ReLU
    - Linear(hidden_dim, output_dim) -> Normalize
    - Output: (B, output_dim) normalized vectors
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 128,
    ):
        """
        Args:
            input_dim: Input dimension (backbone output)
            hidden_dim: Hidden dimension
            output_dim: Output dimension (projection dimension)
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, input_dim)
        
        Returns:
            Normalized projection (B, output_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # Normalize to unit sphere (important for contrastive learning)
        x = F.normalize(x, p=2, dim=1)
        
        return x
