from __future__ import annotations

import torch
from torch import nn


class CachedRGBEncoder(nn.Module):
    """
    Trainable temporal/view fusion on top of frozen cached DINOv2 features.

    Input: [B,K,V,768] for DINOv2-Base.
    """
    def __init__(
        self,
        input_dim: int = 768,
        model_dim: int = 384,
        num_views: int = 4,
        temporal_layers: int = 3,
        temporal_heads: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_views = num_views
        self.proj = nn.Linear(input_dim, model_dim)
        self.view_embed = nn.Parameter(torch.randn(num_views, model_dim) * 0.02)

        self.view_gate = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=temporal_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.pool_query = nn.Parameter(torch.randn(model_dim) * 0.02)

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [B,K,V,D]
        h = self.proj(rgb.float())
        h = h + self.view_embed[None, None, :h.shape[2]]

        view_score = self.view_gate(h).squeeze(-1)
        view_score = torch.softmax(view_score, dim=-1)
        h = torch.einsum("bkv,bkvd->bkd", view_score, h)

        h = self.temporal(h)
        score = torch.einsum("btd,d->bt", h, self.pool_query)
        score = torch.softmax(score, dim=-1)
        pooled = torch.einsum("bt,btd->bd", score, h)
        return pooled, h
