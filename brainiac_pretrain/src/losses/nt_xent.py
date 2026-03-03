"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR.
Also known as InfoNCE loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning.
    
    Given two views z1 and z2 (each shape (B, dim)):
    - Concatenate to (2B, dim)
    - Compute similarity matrix
    - Positive pairs: (i, i+B) and (i+B, i) for i in [0, B-1]
    - Negatives: all other pairs
    - Apply temperature scaling
    - Compute cross-entropy loss
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter (tau) for scaling similarities
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z1: Projected embeddings of first view (B, dim) or (B*K, dim)
            z2: Projected embeddings of second view (B, dim) or (B*K, dim)
        
        Returns:
            Scalar loss value
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Concatenate both views: [z1_1, z1_2, ..., z1_B, z2_1, z2_2, ..., z2_B]
        z = torch.cat([z1, z2], dim=0)  # (2B, dim) or (2B*K, dim)
        
        # Normalize (should already be normalized, but ensure)
        z = F.normalize(z, dim=1)
        
        # Compute similarity matrix: sim[i, j] = z[i] @ z[j]
        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)
        
        # Create labels for positive pairs
        # For sample i in [0, B-1]: positive is i+B
        # For sample i in [B, 2B-1]: positive is i-B
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # (2B,)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        # For each sample i, we want high similarity with its positive pair (label[i])
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


def compute_nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standalone function to compute NT-Xent loss.
    
    Args:
        z1: Projected embeddings of first view (B, dim)
        z2: Projected embeddings of second view (B, dim)
        temperature: Temperature parameter
    
    Returns:
        Scalar loss value
    """
    loss_fn = NTXentLoss(temperature=temperature)
    return loss_fn(z1, z2)
