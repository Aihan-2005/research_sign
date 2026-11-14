
from __future__ import annotations

import numpy as np
import torch

from persian_sea.translation.boundary import (
    BoundaryConfig,
    SentenceBoundaryHead,
    _evaluate,
    predict_windows_with_info,
)
from persian_sea.translation.signshift import (
    count_kl_loss,
    temporal_smoothness_loss,
)


def _record(n=25, fused_dim=16, local_dim=12):
    times = np.arange(n, dtype=np.float32)
    return {
        "video_id": "synthetic",
        "features": np.random.default_rng(4).normal(size=(n, fused_dim)).astype(np.float32),
        "starts": times,
        "ends": times + .75,
        "signer": np.ones((n, local_dim), dtype=np.float32),
        "hands": np.zeros((n, local_dim), dtype=np.float32),
        "face": np.full((n, local_dim), 0.5, dtype=np.float32),
        "labels": np.zeros((n, 2), dtype=np.float32),
        "valid_cues": 5,
        "reference_intervals": [(0., 4.75), (5., 9.75), (10., 14.75),
                                (15., 19.75), (20., 24.75)],
    }


def test_boundary_head_with_tdm_streams_scp_updates_parameters():
    cfg = BoundaryConfig(
        use_tdm=True, use_multistream=True, use_scp=True,
        hidden_dim=16, num_heads=4, num_layers=1, dropout=0.0,
        max_sentence_count=30, max_sentence_segments=12,
    ).validate()
    sample = _record()
    model = SentenceBoundaryHead(
        16, 16, 1, 4, 0.0,
        use_tdm=True, use_multistream=True, use_scp=True,
        local_dim=12, max_sentence_count=30, scp_conv_layers=2,
    )
    x = torch.tensor(sample["features"], requires_grad=True)
    starts = torch.from_numpy(sample["starts"])
    ends = torch.from_numpy(sample["ends"])
    views = {name:torch.from_numpy(sample[name])
             for name in ("signer", "hands", "face")}
    outputs = model(x, starts, ends, streams=views, return_aux=True)
    assert outputs["boundary_logits"].shape == (25, 2)
    assert outputs["count_logits"].shape == (30,)
    loss = (outputs["boundary_logits"].square().mean()
            + 0.15 * temporal_smoothness_loss(outputs["boundary_logits"])
            + 0.1 * count_kl_loss(outputs["count_logits"], 5))
    loss.backward()
    assert x.grad is not None
    assert model.local_fusion.raw_gates.grad is not None
    assert model.scp.classifier[-1].weight.grad is not None

    model.eval()
    windows, info = predict_windows_with_info(
        model, sample, cfg, torch.device("cpu")
    )
    assert info is not None
    assert len(windows) == info.selected_count
    assert windows[0]["start_index"] == 0
    assert windows[-1]["end_index"] == 24
    scores = _evaluate(model, [sample], torch.device("cpu"), .35, cfg)
    assert all(key in scores for key in (
        "weak_f1_10", "weak_f1_25", "weak_f1_50", "weak_ser",
        "weak_count_mae", "weak_scp_count_mae",
    ))


def test_old_boundary_head_still_accepts_three_inputs():
    model = SentenceBoundaryHead(16, 16, 1, 4, 0.0)
    sample = _record()
    logits = model(
        torch.from_numpy(sample["features"]),
        torch.from_numpy(sample["starts"]),
        torch.from_numpy(sample["ends"]),
    )
    assert logits.shape == (25, 2)
    assert model.global_tdm is None
    assert model.local_fusion is None
    assert model.scp is None


def test_cached_local_features_use_view_names_not_view_positions(tmp_path, monkeypatch):
    from persian_sea.translation import boundary

    n, frames, dim = 4, 2, 6
    rgb = np.zeros((n, frames, 4, dim), dtype=np.float32)
    # Deliberately permuted: face, left_hand, signer, right_hand.
    rgb[:, :, 0, :] = 4.
    rgb[:, :, 1, :] = 1.
    rgb[:, :, 2, :] = 2.
    rgb[:, :, 3, :] = 3.
    source = tmp_path / 'visual'
    target = tmp_path / 'boundary'
    source.mkdir()
    np.savez_compressed(
        source / 'demo.npz',
        pose=np.zeros((n, frames, 75, 4), dtype=np.float32),
        rgb=rgb,
        starts=np.arange(n, dtype=np.float32),
        ends=np.arange(n, dtype=np.float32) + .8,
        views=np.asarray(['face', 'left_hand', 'signer', 'right_hand']),
    )

    class FakeVisual(torch.nn.Module):
        def forward(self, pose, rgb):
            fused = rgb.mean(dim=(1, 2, 3), keepdim=False)
            return {'fused': fused.unsqueeze(-1).repeat(1, 8)}

    class FakeVLP:
        def __init__(self):
            self.video = type('VideoTower', (), {'visual': FakeVisual()})()

    class FakeCache:
        def __init__(self, capacity):
            pass

        def get(self, path):
            with np.load(path, allow_pickle=False) as data:
                return {key: np.asarray(data[key])
                        for key in ('pose', 'rgb', 'starts', 'ends')}

    monkeypatch.setattr(boundary, 'VisualCacheLRU', FakeCache)
    result = boundary.extract_segment_features(
        'demo', FakeVLP(), source, target, 'test_hash',
        torch.device('cpu'), batch_size=2, include_local_views=True,
    )
    assert np.allclose(result['face'], 4.)
    assert np.allclose(result['signer'], 2.)
    assert np.allclose(result['hands'], 2.)
    assert result['features'].shape == (n, 8)
    second = boundary.extract_segment_features(
        'demo', FakeVLP(), source, target, 'test_hash',
        torch.device('cpu'), batch_size=2, include_local_views=True,
    )
    assert np.array_equal(result['face'], second['face'])
