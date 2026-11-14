from __future__ import annotations

import numpy as np
import pytest

from persian_sea.alignment.dp import DPConfig, align_monotonic_dp
from persian_sea.io.subtitles import SubtitleCue
from persian_sea.translation.boundary import make_boundary_targets
from persian_sea.types import SignSegment


def test_signshift_soft_prior_can_prefer_high_confidence_visual_end():
    cue = SubtitleCue(0.2, 2.2, "یک جمله")
    signs = [
        SignSegment(0.0, 1.0),
        SignSegment(1.0, 2.0),
        SignSegment(2.0, 3.0),
        SignSegment(3.0, 4.0),
    ]
    cfg = DPConfig(
        duration_penalty_weight=1.0,
        gap_penalty_weight=0.0,
        candidate_window_size=4,
    )
    baseline = align_monotonic_dp([cue], signs, cfg)
    hybrid = align_monotonic_dp(
        [cue], signs, cfg,
        boundary_start_probs=np.array([0.99, 0.10, 0.10, 0.10]),
        boundary_end_probs=np.array([0.10, 0.01, 0.99, 0.10]),
        boundary_weight=5.0,
    )
    assert baseline[0].last_sign_index == 1
    assert hybrid[0].last_sign_index == 2


def test_signshift_prior_rejects_wrong_number_of_scores():
    cue = SubtitleCue(0.0, 2.0, "متن")
    signs = [SignSegment(0.0, 1.0), SignSegment(1.0, 2.0)]
    with pytest.raises(ValueError, match="shape"):
        align_monotonic_dp(
            [cue], signs, DPConfig(candidate_window_size=2),
            boundary_start_probs=np.array([0.9]),
            boundary_end_probs=np.array([0.9]),
            boundary_weight=1.0,
        )


def test_boundary_targets_skip_cues_with_no_actual_visual_overlap():
    starts = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    ends = np.array([0.9, 1.9, 2.9], dtype=np.float32)
    cues = [
        SubtitleCue(10.0, 11.0, "خارج از ویدیو"),
        SubtitleCue(0.1, 1.8, "داخل ویدیو"),
    ]
    labels, valid_cues = make_boundary_targets(starts, ends, cues)
    assert valid_cues == 1
    assert labels[0, 0] == 1.0
    assert labels[1, 1] == 1.0
    assert labels.sum() == 2.0
