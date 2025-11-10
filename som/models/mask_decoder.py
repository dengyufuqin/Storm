import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from .common import LayerNorm2d


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
              activation: str = "relu") -> nn.Sequential:
    layers = []
    dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < num_layers - 1:
            act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
            layers.append(act)
    return nn.Sequential(*layers)


class MaskDecoderSimple(nn.Module):
    """
    Decoder that generates masks and IoU predictions.
    Uses a parallel hypernetwork with LayerNorm and LayerScale for stability.
    """
    def __init__(
        self,
        transformer_dim: int,
        num_multimask_outputs: int = 1,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        layer_scale_init: float = 1e-3,
    ):
        super().__init__()
        # number of masks
        self.num_masks = num_multimask_outputs
        # upscaling backbone
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, 2, 2),
            LayerNorm2d(transformer_dim // 4), nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, 2, 2),
            LayerNorm2d(transformer_dim // 8), nn.GELU(),
        )
        Cp = transformer_dim // 8
        K = self.num_masks

        # parallel hypernetwork: outputs Cp*K dims
        self.hyper_net = _make_mlp(transformer_dim, transformer_dim, Cp * K, num_layers=3)
        self.hyper_norm = nn.LayerNorm(Cp * K)
        # LayerScale parameter
        self.gamma = nn.Parameter(torch.ones(1) * layer_scale_init)

        # skip connection to predict mask from upscaled features
        self.skip_proj = nn.Conv2d(Cp, K, kernel_size=1)

        # refinement head
        self.refine_head = nn.Sequential(
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(K, K, kernel_size=3, padding=1),
        )

        # IoU head
        self.iou_head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Dropout(0.1),
            *_make_mlp(transformer_dim, iou_head_hidden_dim, K, iou_head_depth)
        )

    def forward(
        self,
        query_emb: Tensor,
        multimask_output: bool = False,
        lang_ctx: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          query_emb:        (B, C, h, w)
          multimask_output: whether to return multiple masks
        Returns:
          masks:    (B, K, h', w')
          iou_preds: (B, K)
        """
        B, C, h, w = query_emb.shape
        K = self.num_masks

        # IoU predictions
        pooled = query_emb.flatten(2).mean(-1)    # [B, C]
        iou_preds = self.iou_head(pooled)         # [B, K]

        # Upscale features
        up = self.output_upscaling(query_emb)     # [B, Cp, hp, wp]
        Bp, Cp, hp, wp = up.shape
        # flatten spatial dims
        up_flat = up.view(Bp, Cp, hp * wp)       # [B, Cp, S]

        # Hypernetwork generates all kernel weights at once
        kernels = self.hyper_net(pooled)         # [B, Cp*K]
        kernels = self.hyper_norm(kernels)
        kernels = kernels.view(B, K, Cp)         # [B, K, Cp]

        # compute dynamic mask logits: batched einsum
        dynamic_masks = torch.einsum('bkp,bps->bks', kernels, up_flat).contiguous()  # [B, K, S]
        dynamic_masks = dynamic_masks.view(B, K, hp, wp)

        # compute skip mask logits
        skip_masks = self.skip_proj(up.contiguous()) # [B, K, hp, wp]

        # combine with LayerScale and refine
        masks = skip_masks + self.gamma * dynamic_masks
        masks = self.refine_head(masks)

        if not multimask_output:
            masks = masks[:, :1]
            iou_preds = iou_preds[:, :1]
        return masks, iou_preds
