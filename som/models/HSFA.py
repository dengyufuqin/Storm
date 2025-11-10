import torch
import torch.nn as nn
from typing import Optional
from .common import AdaNormZero, MLPBlock, Attention
import torch
import torch.nn as nn
from typing import Optional
from .common import AdaNormZero, MLPBlock, Attention

class HSFABlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_downsample_rate: int = 1,
        cond_dim: Optional[int] = None,
        mlp_dim: int = 1536,
        drop_path_rate: float = 0.0,
        num_views: int = 1,
        patch_dim: int = 1,
        layer_scale_init_value: float = 1e-4,   
    ):
        super().__init__()
        self.num_views = num_views
        self.patch_dim = patch_dim

        # 1) Query 
        self.norm_q1 = nn.LayerNorm(embedding_dim)
        self.norm_k1 = nn.LayerNorm(embedding_dim)
        self.query_self_attn = Attention(embedding_dim, num_heads)

        # LayerScale 
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(embedding_dim),
            requires_grad=True
        )
        self.query_norm1 = AdaNormZero(embedding_dim, cond_dim)

        # 2) Cross-attention
        self.norm_q2 = nn.LayerNorm(embedding_dim)
        self.norm_k2 = nn.LayerNorm(embedding_dim)
        self.cross_attn = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(embedding_dim),
            requires_grad=True
        )
        self.query_norm2 = AdaNormZero(embedding_dim, cond_dim)

        # 3) MLP + AdaNormZero
        self.mlp = MLPBlock(embedding_dim, mlp_dim)
        self.layer_scale_3 = nn.Parameter(
            layer_scale_init_value * torch.ones(embedding_dim),
            requires_grad=True
        )
        self.query_norm3 = AdaNormZero(embedding_dim, cond_dim)

        # 4) Reference view-level
        self.norm_q4 = nn.LayerNorm(embedding_dim)
        self.norm_k4 = nn.LayerNorm(embedding_dim)
        self.ref_view_self_attn = Attention(embedding_dim, num_heads)
        self.layer_scale_4 = nn.Parameter(
            layer_scale_init_value * torch.ones(embedding_dim),
            requires_grad=True
        )

        # 5) reference
        self.norm_q5 = nn.LayerNorm(embedding_dim)
        self.norm_k5 = nn.LayerNorm(embedding_dim)
        self.ref_global_self_attn = Attention(embedding_dim, num_heads)
        self.layer_scale_5 = nn.Parameter(
            layer_scale_init_value * torch.ones(embedding_dim),
            requires_grad=True
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        ref_tokens: torch.Tensor,
        pos_q: torch.Tensor,
        pos_ref: torch.Tensor,
        lang_ctx: Optional[torch.Tensor] = None,
    ):
        B, Nq, C = query_tokens.shape
        _, Nv, _ = ref_tokens.shape
        V, M = self.num_views, self.patch_dim * self.patch_dim

        # --- 1) Query ---
        q_in = query_tokens + pos_q
        # QKNorm
        q_normed = self.norm_q1(q_in)
        k_normed = self.norm_k1(q_in)
        sa = self.query_self_attn(q_normed, k_normed, query_tokens)
        # LayerScale 
        query_tokens = query_tokens + sa * self.layer_scale_1
        query_tokens = self.query_norm1(query_tokens, lang_ctx)

        # --- 2) Cross-attention ---
        q_in2 = query_tokens + pos_q
        k_in2 = ref_tokens + pos_ref
        qn = self.norm_q2(q_in2)
        kn = self.norm_k2(k_in2)
        ca = self.cross_attn(qn, kn, ref_tokens)
        query_tokens = query_tokens + ca * self.layer_scale_2
        query_tokens = self.query_norm2(query_tokens, lang_ctx)

        # --- 3) MLP ---
        mlp_out = self.mlp(query_tokens)
        query_tokens = query_tokens + mlp_out * self.layer_scale_3
        query_tokens = self.query_norm3(query_tokens, lang_ctx)

        # --- 4) Per-view  on ref ---
        refs = ref_tokens.view(B, V, M, C)
        if pos_q is not None:
            refs_in = (refs + pos_q.unsqueeze(1)).reshape(B*V, M, C)
        else:
            refs_in = refs.reshape(B*V, M, C)
        qv = self.norm_q4(refs_in)
        kv = self.norm_k4(refs_in)
        refs_sa = self.ref_view_self_attn(qv, kv, refs.reshape(B*V, M, C))
        refs = (refs.reshape(B*V, M, C) + refs_sa * self.layer_scale_4).view(B, V, M, C)
        ref_cat = refs.view(B, Nv, C)

        # --- 5)  ref  ---
        ref_in = ref_cat + pos_ref
        qg = self.norm_q5(ref_in)
        kg = self.norm_k5(ref_in)
        ref_sa = self.ref_global_self_attn(qg, kg, ref_cat)
        ref_out = ref_cat + ref_sa * self.layer_scale_5

        return query_tokens, ref_out

class HSFATransformer(nn.Module):
    """
    Hierarchic Spatial Fusion Attention Transformer.
    Iteratively updates query and reference embeddings with HSFA blocks,
    injecting spatial pos embeddings and optional language context.
    """
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        attention_downsample_rate: int = 1,
        cond_dim: Optional[int] = None,
        mlp_dim: int = 1536,
        num_views: int = 1,
        patch_dim: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            HSFABlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                attention_downsample_rate=attention_downsample_rate,
                cond_dim=cond_dim,
                mlp_dim=mlp_dim,
                num_views=num_views,
                patch_dim=patch_dim,
            ) for _ in range(depth)]
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        ref_emb: torch.Tensor,
        lang_ctx: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, torch.Tensor):
        # query_emb: [B, C, P, P]; ref_emb: [B, V, C, P, P]
        B, C, P, _ = query_emb.shape
        V = ref_emb.shape[1]
        Nq = P * P
        Nv = V * Nq

        # flatten embeddings to sequences
        q_tokens = query_emb.view(B, C, Nq).permute(0, 2, 1)   # [B, Nq, C]
        ref_tokens = ref_emb.permute(0, 1, 3, 4, 2) # [B, V, P, P, C]

        ref_tokens = ref_tokens.reshape(B, V * P * P, C)  # [B, N_v, C]

        # prepare positional tokens
        if pos_emb is not None:
            pos_q = pos_emb.view(B, C, Nq).permute(0, 2, 1)    # [B, Nq, C]
            # replicate for each view
            # pos_ref = pos_q.repeat(1, V, 1)                   # [B, Nv, C]
            pos_ref = (
                pos_q
                .unsqueeze(1)             # [B, 1, Nq, C]
                .expand(-1, V, -1, -1)    # [B, V, Nq, C]
                .reshape(B, V * Nq, C)    # [B, Nv, C]
            )
        else:
            pos_q = pos_ref = None

        # iterative HSFA
        for layer in self.layers:
            q_tokens, ref_tokens = layer(
                q_tokens, 
                ref_tokens,
                pos_q, 
                pos_ref,
                lang_ctx,
            )

        # reshape back to spatial maps
        # query_out = q_tokens.permute(0, 2, 1).view(B, C, P, P)
        # ref_out = ref_tokens.view(B, V, C, P, P)
        query_out = q_tokens.permute(0, 2, 1).view(B, C, P, P)
        # ref_out = ref_tokens.reshape(B, V, P, P, C).permute(0, 1, 4, 2, 3)
        ref_out = ref_tokens.view(B, V, P, P, C)       # [B, V, P, P, C]
        ref_out = ref_out.permute(0, 1, 4, 2, 3)           # [B, V, C, P, P]
        return query_out, ref_out
