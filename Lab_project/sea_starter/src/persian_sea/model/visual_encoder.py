from __future__ import annotations

import torch
from torch import nn


from persian_sea.model.pose_encoder import PoseSemanticEncoder
from persian_sea.model.rgb_encoder import CachedRGBEncoder
from persian_sea.model.fusion import GatedPoseRGBFusion


class PoseRGBVisualEncoder(nn.Module):
    def __init__(
        self,
        pose_input_dim: int = 4,
        rgb_input_dim: int = 768,
        model_dim: int = 384,
        output_dim: int = 512,
        num_views: int = 4,
    ):
        super().__init__()
        self.pose = PoseSemanticEncoder(
            input_dim=pose_input_dim, model_dim=model_dim,
        )
        self.rgb = CachedRGBEncoder(
            input_dim=rgb_input_dim,
            model_dim=model_dim,
            num_views=num_views,
        )
        self.fusion = GatedPoseRGBFusion(model_dim, output_dim)

    def forward(self, pose: torch.Tensor, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        pose_vec, pose_tokens = self.pose(pose)
        rgb_vec, rgb_tokens = self.rgb(rgb)
        fused = self.fusion(pose_vec, rgb_vec)
        return {
            "fused": fused,
            "pose": pose_vec,
            "rgb": rgb_vec,
            "pose_tokens": pose_tokens,
            "rgb_tokens": rgb_tokens,
        }
