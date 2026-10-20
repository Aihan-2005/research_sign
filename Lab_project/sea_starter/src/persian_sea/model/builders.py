
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

from persian_sea.core.registry import Registry


class PoseOnlyVisualEncoder(nn.Module):

    def __init__(self, output_dim: int, model_dim: int = 384, **kwargs):
        super().__init__()
        from persian_sea.model.pose_encoder import PoseSemanticEncoder
        self.pose = PoseSemanticEncoder(input_dim=4, model_dim=model_dim)
        self.projection = nn.Linear(model_dim, output_dim)

    def forward(self, pose: torch.Tensor, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled, tokens = self.pose(pose)
        return {"fused": F.normalize(self.projection(pooled), dim=-1),
                "pose": pooled, "pose_tokens": tokens}


def _build_pose_rgb(**kwargs):
    from persian_sea.model.visual_encoder import PoseRGBVisualEncoder
    return PoseRGBVisualEncoder(**kwargs)


visual_backends = Registry("visual encoder")
visual_backends.register("pose_rgb", _build_pose_rgb)
visual_backends.register("pose_only", lambda **kwargs: PoseOnlyVisualEncoder(**kwargs))


def _build_pose_rgb_motion(**kwargs):
    from persian_sea.model.motion_encoder import PoseRGBMotionVisualEncoder
    return PoseRGBMotionVisualEncoder(**kwargs)


visual_backends.register("pose_rgb_motion", _build_pose_rgb_motion)


def build_visual_encoder(backend: str, **kwargs):
    return visual_backends.create(backend, **kwargs)
