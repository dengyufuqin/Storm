import torch
import torch.nn as nn
from typing import Tuple
from .query_encoder import Dinov2Encoder

class PromptEncoderMultiViewSimple(nn.Module):
    """
    Simplified PromptEncoder for multi-view inputs: uses a DINOv2 encoder
    to extract per-view patch embeddings without fusion, preserving each view's embedding
    for HSFA processing.

    Returns:
      ref_embeddings: Tensor[B, N_views, embed_dim, patch_dim, patch_dim]
    """
    def __init__(
        self,
        encoder_args: dict,
        device: str,
    ):
        super().__init__()
        # Use the simplified DINOv2 image encoder
        self.view_encoder = Dinov2Encoder(
            **encoder_args,
            device=device
        )

    def forward(self, view_imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
          view_imgs: Tensor[B, N_views, 3, H, W]
        Returns:
          ref_embeddings: Tensor[B, N_views, D, P, P]
        """
        B, N, C, H, W = view_imgs.shape
        # Flatten views into batch dimension
        views_flat = view_imgs.view(B * N, C, H, W)
        # Encode each view -> [B*N, D, P, P]
        patch_feats = self.view_encoder(views_flat)
        # Extract dims
        _, D, P, _ = patch_feats.shape
        # Reshape back to multi-view tensor: [B, N, D, P, P]
        ref_embeddings = patch_feats.view(B, N, D, P, P)
        return ref_embeddings
    