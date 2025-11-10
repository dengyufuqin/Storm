# models/som_multiview.py

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from .query_encoder import Dinov2Encoder
from .reference_encoder import PromptEncoderMultiViewSimple
from .HSFA import HSFATransformer
from .mask_decoder import MaskDecoderSimple
from .common import PositionEmbeddingSine

class SOMMultiview(nn.Module):
    """
    Simplified Object Masking (SOM) model for multi-view reference inputs.

    Pipeline:
      1) Encode query image via DINOv2 -> patch embedding [B, C, P, P]
      2) Encode reference views via DINOv2 -> per-view patch embeddings [B, V, C, P, P]
      3) HSFA fusion: iterative spatial fusion of query and reference embeddings (with optional pos and lang ctx)
      4) Decode mask(s) from fused query embedding
    """
    def __init__(
        self,
        query_encoder_args: Dict[str, Any],
        prompt_encoder_args: Dict[str, Any],
        hsfa_config: Dict[str, Any],
        decoder_args: Dict[str, Any],
        device: str,
    ):
        super().__init__()
        # 1) Query encoder: DINOv2 patch embeddings
        self.query_encoder = Dinov2Encoder(
            **query_encoder_args,
            device=device
        )
        # 2) Reference encoder: per-view DINOv2 embeddings
        self.prompt_encoder = PromptEncoderMultiViewSimple(
            **prompt_encoder_args,
            device=device
        )

        # 3) Positional encoding for query (and optionally ref)
        embedding_dim = self.query_encoder.embed_dim
        self.pos_encoder = PositionEmbeddingSine(
            num_pos_feats=embedding_dim // 2,
            normalize=True
        )

        # 4) HSFA transformer for spatial fusion
        patch_dim = self.query_encoder.patch_dim
        self.hsfa = HSFATransformer(
            depth=hsfa_config['depth'],
            embedding_dim=embedding_dim,
            num_heads=hsfa_config['num_heads'],
            mlp_dim=hsfa_config['mlp_dim'],
            attention_downsample_rate=hsfa_config.get('attention_downsample_rate', 1),
            cond_dim=hsfa_config.get('cond_dim', None),
            num_views=hsfa_config['num_views'],
            patch_dim=patch_dim,
        )

        # 5) Mask decoder: upsampling + hypernet
        self.decoder = MaskDecoderSimple(
            transformer_dim=embedding_dim,
            num_multimask_outputs=decoder_args.get('num_multimask_outputs', 1),
            iou_head_depth=decoder_args.get('iou_head_depth', 3),
            iou_head_hidden_dim=decoder_args.get(
                'iou_head_hidden_dim', embedding_dim
            ),
        )

    def forward(
        self,
        query_img: torch.Tensor,
        view_imgs: torch.Tensor,
        multimask_output: bool = False,
        lang_ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          query_img:       Tensor[B,3,H,W]
          view_imgs:       Tensor[B,N_views,3,H,W]
          multimask_output: bool, whether to return multiple masks
          lang_ctx:        Optional language context [B, cond_dim]

        Returns:
          masks:     Tensor[B, K, h', w']
          iou_preds: Tensor[B, K]
        """
        # 1) Encode query and refs
        query_emb = self.query_encoder(query_img)    # [B, C, P, P]
        ref_emb = self.prompt_encoder(view_imgs)     # [B, V, C, P, P]

        # 2) Positional encoding
        pos_emb = self.pos_encoder(query_emb)        # [B, C, P, P]

        # 3) HSFA spatial fusion (inject pos and optional lang_ctx)
        query_fused, ref_fused = self.hsfa(
            query_emb,
            ref_emb,
            lang_ctx=lang_ctx,
            pos_emb=pos_emb
        )

        # 4) Decode mask(s) from fused query embedding
        masks, iou_preds = self.decoder(
            query_fused,
            multimask_output=multimask_output,
            lang_ctx=lang_ctx,
        )
        return masks, iou_preds
