from __future__ import annotations

import torch
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
        return torch.nn.functional.normalize(fused, dim=-1)
