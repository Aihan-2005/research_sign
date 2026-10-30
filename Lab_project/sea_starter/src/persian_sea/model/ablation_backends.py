
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_positions(length: int, dim: int, *, device, dtype) -> torch.Tensor:
    if length < 1 or dim < 1:
        raise ValueError('length and dim must be positive')
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                      * (-math.log(10000.0) / dim))
    values = torch.zeros(length, dim, device=device, dtype=torch.float32)
    values[:, 0::2] = torch.sin(position * scale)
    if dim > 1:
        values[:, 1::2] = torch.cos(position * scale[:dim//2])
    return values.to(dtype=dtype)


class TemporalPositionSignal(nn.Module):

    def __init__(self, scale: float = 0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('Expected [batch, frames, channels]')
        pe = sinusoidal_positions(x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
        return x + self.scale * pe.unsqueeze(0)


def _attach_position(temporal: nn.Module, scale: float = 0.1):
    signal = TemporalPositionSignal(scale)

    temporal.add_module('intra_segment_position', signal)
    def hook(module, args):
        return (module.intra_segment_position(args[0]), *args[1:])
    temporal.register_forward_pre_hook(hook)


class RGBOnlyVisualEncoder(nn.Module):
    def __init__(self, output_dim: int, model_dim: int = 384,
                 rgb_input_dim: int = 768, num_views: int = 4, **kwargs):
        super().__init__()
        from persian_sea.model.rgb_encoder import CachedRGBEncoder
        self.rgb = CachedRGBEncoder(input_dim=rgb_input_dim, model_dim=model_dim,
                                    num_views=num_views)
        self.projection = nn.Linear(model_dim, output_dim)

    def forward(self, pose: torch.Tensor, rgb: torch.Tensor):
        pooled, tokens = self.rgb(rgb)
        return {'fused': F.normalize(self.projection(pooled), dim=-1),
                'rgb': pooled, 'rgb_tokens': tokens}


class PoseRGBPositionalVisualEncoder(nn.Module):
    def __init__(self, pose_input_dim: int = 4, rgb_input_dim: int = 768,
                 model_dim: int = 384, output_dim: int = 512,
                 num_views: int = 4, **kwargs):
        super().__init__()
        from persian_sea.model.visual_encoder import PoseRGBVisualEncoder
        self.base = PoseRGBVisualEncoder(pose_input_dim=pose_input_dim,
                rgb_input_dim=rgb_input_dim, model_dim=model_dim,
                output_dim=output_dim, num_views=num_views)
        _attach_position(self.base.pose.temporal)
        _attach_position(self.base.rgb.temporal)

    def forward(self, pose: torch.Tensor, rgb: torch.Tensor):
        return self.base(pose, rgb)
