import pytest
import torch

pytest.importorskip("torch")

from persian_sea.model.builders import build_visual_encoder


def test_pose_rgb_cross_attention_backend_forward():
    model = build_visual_encoder(
        "pose_rgb_cross_attention",
        pose_input_dim=4,
        rgb_input_dim=12,
        num_views=2,
        model_dim=48,
        output_dim=16,
    )
    model.eval()
    pose = torch.randn(2, 4, 75, 4)
    rgb = torch.randn(2, 4, 2, 12)
    with torch.no_grad():
        output = model(pose, rgb)
    assert output["fused"].shape == (2, 16)
    assert output["fused_tokens"].shape == (2, 4, 48)
    assert torch.isfinite(output["fused"]).all()
