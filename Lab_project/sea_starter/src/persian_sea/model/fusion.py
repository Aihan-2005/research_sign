from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


class GatedPoseRGBFusion(nn.Module):
    def __init__(self, dim: int = 384, out_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, pose: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([pose, rgb], dim=-1))
        fused = gate * rgb + (1.0 - gate) * pose
        fused = self.out(fused)
        return F.normalize(fused, dim=-1)


def _sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    """Create an order signal for the within-segment sample axis."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(dim, 1))
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * scale)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * scale[:pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


class BidirectionalPoseRGBCrossAttentionFusion(nn.Module):
    """Optional temporal cross-modal fusion over aligned Pose/RGB sample tokens.

    Pose tokens query RGB tokens and RGB tokens query Pose tokens. The module
    keeps the original gated-fusion backend intact; this is a separate ablation.
    """

    def __init__(self, dim: int = 384, out_dim: int = 512,
                 num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.pose_to_rgb = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.rgb_to_pose = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.pose_norm = nn.LayerNorm(dim)
        self.rgb_norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid(),
        )
        self.output = nn.Sequential(nn.Linear(dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, pose_tokens: torch.Tensor,
                rgb_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pose_tokens.ndim != 3 or rgb_tokens.ndim != 3:
            raise ValueError("Pose/RGB tokens must have shape [B,K,D]")
        if pose_tokens.shape != rgb_tokens.shape:
            raise ValueError(
                f"Pose/RGB token shapes must match, got {pose_tokens.shape} and {rgb_tokens.shape}"
            )
        if pose_tokens.shape[1] < 1:
            raise ValueError("Cross-attention requires at least one temporal token")

        position = _sinusoidal_positions(
            pose_tokens.shape[1], pose_tokens.shape[2],
            pose_tokens.device, pose_tokens.dtype,
        ).unsqueeze(0)
        pose = pose_tokens + position
        rgb = rgb_tokens + position

        pose_context, _ = self.pose_to_rgb(
            pose, rgb, rgb, need_weights=False,
        )
        rgb_context, _ = self.rgb_to_pose(
            rgb, pose, pose, need_weights=False,
        )
        pose = self.pose_norm(pose + pose_context)
        rgb = self.rgb_norm(rgb + rgb_context)

        gate = self.gate(torch.cat([pose, rgb], dim=-1))
        fused_tokens = gate * rgb + (1.0 - gate) * pose
        pooled = fused_tokens.mean(dim=1)
        fused = F.normalize(self.output(pooled), dim=-1)
        return fused, fused_tokens
