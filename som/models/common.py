
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Type
import math

class MLPBlock(nn.Module):
    """
    Simple feed-forward MLP block: Linear -> Activation -> Linear
    """
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """
    LayerNorm over channels for 2D feature maps.
    Implements per-channel normalization: normalize across H, W per sample.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x_norm + self.bias.view(1, -1, 1, 1)


class AdaNormZero(nn.Module):
    """
    Zero-centered Adaptive Normalization (AdaNorm-Zero)

    Per-feature normalization with learnable scale (gamma) and shift (delta),
    optionally conditioned on a context vector.
    """
    def __init__(
        self,
        dim: int,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.delta = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-5
        if cond_dim is not None:
            self.ctx_proj = nn.Sequential(
                nn.Linear(cond_dim, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim * 2),
            )
        else:
            self.ctx_proj = None

    def forward(self, x: Tensor, ctx: Optional[Tensor] = None) -> Tensor:
        # x: [B, N, C]
        # Standard per-feature normalize over last dim
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # base gamma and delta
        gamma = self.gamma.view(1, 1, -1)
        delta = self.delta.view(1, 1, -1)
        # context conditioning
        if self.ctx_proj is not None and ctx is not None:
            # ctx: [B, cond_dim]
            gd = self.ctx_proj(ctx)  # [B, 2*C]
            g_ctx, d_ctx = gd.chunk(2, dim=-1)
            gamma = gamma + g_ctx.unsqueeze(1)
            delta = delta + d_ctx.unsqueeze(1)
        return x_norm * gamma + delta



class Attention(nn.Module):
    """
    Multi-head attention with optional internal downsampling,
    now using FlashAttention under the hood.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim  = embedding_dim // downsample_rate
        self.num_heads     = num_heads
        assert self.internal_dim % num_heads == 0, \
            f"internal_dim {self.internal_dim} not divisible by num_heads {num_heads}"
        self.head_dim = self.internal_dim // num_heads

        self.q_proj   = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj   = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj   = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, C // self.num_heads) \
                .permute(0, 2, 1, 3)  # [B, heads, N, head_dim]

    def _recombine_heads(self, x: Tensor) -> Tensor:
        B, H, N, D = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, N, H * D)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # 1) project
        qh = self._separate_heads(self.q_proj(q))  # [B, heads, Nq, head_dim]
        kh = self._separate_heads(self.k_proj(k))  # [B, heads, Nk, head_dim]
        vh = self._separate_heads(self.v_proj(v))  # [B, heads, Nk, head_dim]

        # 2) Flash‐optimized scaled dot‐product attention
        #    (handles the 1/sqrt(d) scaling internally)
        attn_out = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )  # [B, heads, Nq, head_dim]

        # 3) recombine heads & final projection
        out = self._recombine_heads(attn_out)      # [B, Nq, internal_dim]
        return self.out_proj(out)                  # [B, Nq, embedding_dim]


class FlashAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, downsample_rate=1, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim  = embedding_dim // downsample_rate
        self.num_heads     = num_heads
        assert self.internal_dim % num_heads == 0
        self.head_dim = self.internal_dim // num_heads

        self.q_proj  = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj  = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj  = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout = dropout

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, Nq, _ = q.shape
        Nk = k.size(1)

        def proj_and_split(x, proj, N):
            x = proj(x).view(B, N, self.num_heads, self.head_dim)
            return x.permute(0,2,1,3)

        qh = proj_and_split(q, self.q_proj, Nq)
        kh = proj_and_split(k, self.k_proj, Nk)
        vh = proj_and_split(v, self.v_proj, Nk)

        # 走 PyTorch 内置的 flash kernel
        y = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=False
        )
        y = y.permute(0,2,1,3).reshape(B, Nq, self.internal_dim)
        return self.out_proj(y)

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        # create a dummy mask: all valid
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=128, max_h=50, max_w=50):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, num_pos_feats)
        self.col_embed = nn.Embedding(max_w, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        # embeddings for each row/col index
        cols = torch.arange(W, device=x.device)
        rows = torch.arange(H, device=x.device)
        x_emb = self.col_embed(cols)  # [W, num_pos_feats]
        y_emb = self.row_embed(rows)  # [H, num_pos_feats]
        # build grid [H, W, 2*num_pos_feats]
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1)
        # reshape to [B, 2*num_pos_feats, H, W]
        return pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
