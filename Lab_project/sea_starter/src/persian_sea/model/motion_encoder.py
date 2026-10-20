

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class PoseDeltaEncoder(nn.Module):


    def __init__(self, output_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        
        self.temporal = nn.Sequential(
            nn.Conv1d(27, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 4 or pose.shape[2] != 75 or pose.shape[-1] != 4:
            raise ValueError(f"Expected pose [B,K,75,4], got {tuple(pose.shape)}")
        if pose.shape[1] < 2:
            return pose.new_zeros(pose.shape[0], self.projection.out_features)
        coordinates = pose[..., :3].float()
        confidence = pose[..., 3].float().clamp(0, 1)
        pair_conf = torch.minimum(confidence[:, 1:], confidence[:, :-1])
        delta = coordinates[:, 1:] - coordinates[:, :-1]
        parts = ((0, 33), (33, 54), (54, 75))
        features = []
        energies = []
        for first, last in parts:
            d = delta[:, :, first:last]
            valid = (pair_conf[:, :, first:last] > 0.05).to(d.dtype)[..., None]
            count = valid.sum(dim=2).clamp_min(1.0)
            signed = (d * valid).sum(dim=2) / count
            absolute = (d.abs() * valid).sum(dim=2) / count
            rms = torch.sqrt(((d ** 2) * valid).sum(dim=2) / count + 1e-12)
            features.extend((signed, absolute, rms))
            energies.append(absolute.mean(dim=(1, 2)))
        x = torch.cat(features, dim=-1).transpose(1, 2)
        pooled = self.temporal(x).squeeze(-1)
        # No reliable movement -> neutral motion branch, not a learned bias.
        active = (torch.stack(energies, dim=1).amax(dim=1) > 1e-6)[:, None]
        vector = self.projection(pooled)
        return F.normalize(vector, dim=-1) * active.to(vector.dtype)


class PoseRGBMotionVisualEncoder(nn.Module):


    def __init__(self, output_dim: int, **kwargs):
        super().__init__()
        from persian_sea.model.visual_encoder import PoseRGBVisualEncoder
        self.base = PoseRGBVisualEncoder(output_dim=output_dim, **kwargs)
        self.motion = PoseDeltaEncoder(output_dim=output_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, 1),
        )

    def forward(self, pose: torch.Tensor, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        result = self.base(pose, rgb)
        m = self.motion(pose)
        base = result["fused"]
        weight = torch.sigmoid(self.gate(torch.cat([base, m], dim=-1)))
        output = F.normalize(base + weight * m, dim=-1)
        return {**result, "fused": output, "motion": m}
