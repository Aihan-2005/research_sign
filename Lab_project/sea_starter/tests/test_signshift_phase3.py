from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from persian_sea.translation.signshift import (
    MultiScaleTemporalDifference,
    LocalStreamFusion,
    SegmentCountHead,
    count_kl_loss,
    temporal_smoothness_loss,
    decode_count_constrained,
    sentence_segmentation_metrics,
    aggregate_sentence_metrics,
)


def test_tdm_multiscale_retains_shape_and_gradients():
    module = MultiScaleTemporalDifference(16, (1, 2, 4, 8), dropout=0.0)
    x = torch.randn(5, 16, requires_grad=True)
    y = module(x)
    assert y.shape == (5, 16)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None
    assert module.delta_projection.weight.grad is not None


def test_tdm_handles_short_single_segment():
    module = MultiScaleTemporalDifference(8, (1, 2, 4, 8))
    result = module(torch.randn(1, 8))
    assert result.shape == (1, 8)


def test_multistream_gates_use_three_named_visual_views():
    model = LocalStreamFusion(12, 16, (1, 2), dropout=0.0)
    global_hidden = torch.randn(6, 16, requires_grad=True)
    streams = {name: torch.randn(6, 12) for name in ("signer", "hands", "face")}
    fused = model(global_hidden, streams)
    fused.square().mean().backward()
    assert fused.shape == (6, 16)
    assert model.raw_gates.grad is not None
    with pytest.raises(ValueError, match="face"):
        model(global_hidden, {k:v for k,v in streams.items() if k != "face"})


def test_scp_loss_learns_count_and_backpropagates():
    head = SegmentCountHead(12, max_sentences=15, conv_layers=2, dropout=0.0)
    features = torch.randn(18, 12, requires_grad=True)
    logits = head(features)
    loss = count_kl_loss(logits, count=7, sigma=1.0)
    assert logits.shape == (15,)
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    with pytest.raises(ValueError, match="exceeds"):
        count_kl_loss(logits, count=16, sigma=1.0)


def test_temporal_smoothness_loss():
    a = torch.zeros(5, 2)
    b = torch.tensor([[0., 0.], [5., 5.], [-5., -5.], [5., 5.], [0., 0.]])
    assert temporal_smoothness_loss(a) < temporal_smoothness_loss(b)


def test_exact_count_dp_returns_full_coverage_no_overlap():
    n = 27
    starts = np.arange(n, dtype=np.float32)
    ends = starts + 0.9

    s = np.full(n, 0.05)
    e = np.full(n, 0.05)
    for index in (6, 13, 20):
        s[index] = 0.99
        e[index-1] = 0.99
    counts = np.full(15, -5.0)
    counts[3] = 9.0
    windows, info = decode_count_constrained(
        starts, ends, s, e, counts, min_segments=2, max_segments=9
    )
    assert len(windows) == 4
    assert info.selected_count == 4
    assert [w["start_index"] for w in windows] == [0, 6, 13, 20]
    assert windows[0]["start_index"] == 0
    assert windows[-1]["end_index"] == n-1
    for prev, cur in zip(windows, windows[1:]):
        assert prev["end_index"]+1 == cur["start_index"]


def test_exact_count_dp_explains_count_adjustment():
    starts = np.arange(10, dtype=np.float32)
    ends = starts + 0.9
    ps = np.full(10, 0.5)
    pe = np.full(10, 0.5)
    counts = np.zeros(20)
    counts[14] = 4.0
    windows, info = decode_count_constrained(
        starts, ends, ps, pe, counts,
        min_segments=2, max_segments=5,
    )
    assert info.predicted_count == 15
    assert info.selected_count == 5
    assert len(windows) == 5
    assert info.constrained
    assert "not feasible" in info.adjustment_reason


def test_segmentation_metrics_exact_matches_and_count_error():
    gold = [(0., 2.), (2., 4.), (4., 6.)]
    exact = sentence_segmentation_metrics(gold, gold)
    assert exact["f1_10"] == 1.0
    assert exact["f1_25"] == 1.0
    assert exact["f1_50"] == 1.0
    assert exact["ser"] == 0.0
    assert exact["count_mae"] == 0.0
    missing = sentence_segmentation_metrics(gold[:2], gold)
    assert math.isclose(missing["ser"], 1/3)
    assert missing["count_mae"] == 1
    out = aggregate_sentence_metrics([exact, missing])
    assert out["videos"] == 2
    assert math.isclose(out["count_mae"], 0.5)


def test_count_dp_rejects_nonfinite_and_invalid_timing():
    starts = np.array([0., 1.])
    ends = np.array([1., 2.])
    probs = np.array([0.5, np.nan])
    with pytest.raises(ValueError, match="nonfinite"):
        decode_count_constrained(starts, ends, probs, np.ones(2)*0.5,
                                 np.ones(4), min_segments=1, max_segments=4)


def test_count_dp_single_segment_clip_does_not_crash():
    windows, info = decode_count_constrained(
        np.array([0.0]), np.array([0.6]),
        np.array([0.5]), np.array([0.5]),
        np.array([-1., 2., 0.]),
        min_segments=2, max_segments=32,
    )
    assert len(windows) == 1
    assert windows[0]["start_index"] == 0
    assert info.selected_count == 1
    assert info.constrained
