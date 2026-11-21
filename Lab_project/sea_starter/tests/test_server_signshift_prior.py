import numpy as np
import pytest

from persian_sea.alignment.dp import DPConfig, align_monotonic_dp
from persian_sea.io.subtitles import SubtitleCue
from persian_sea.types import SignSegment
from persian_sea.translation.boundary import make_boundary_targets


def test_prior_changes_temporal_endpoint():
    cue = SubtitleCue(0.2, 2.2, "یک جمله")
    signs = [SignSegment(float(i), float(i+1)) for i in range(4)]
    cfg = DPConfig(duration_penalty_weight=1.0, gap_penalty_weight=0.0,
                   candidate_window_size=4)
    baseline = align_monotonic_dp([cue], signs, cfg)
    hybrid = align_monotonic_dp(
        [cue], signs, cfg,
        boundary_start_probs=np.array([0.99, 0.1, 0.1, 0.1]),
        boundary_end_probs=np.array([0.1, 0.01, 0.99, 0.1]),
        boundary_weight=5.0,
    )
    assert baseline[0].last_sign_index == 1
    assert hybrid[0].last_sign_index == 2


def test_prior_requires_matched_shape_and_ranges():
    signs = [SignSegment(0., 1.), SignSegment(1., 2.)]
    cues = [SubtitleCue(0., 2., "متن")]
    with pytest.raises(ValueError, match="shape"):
        align_monotonic_dp(cues, signs, DPConfig(),
                           boundary_start_probs=np.array([0.9]),
                           boundary_end_probs=np.array([0.9]),
                           boundary_weight=1.0)
    with pytest.raises(ValueError, match="within"):
        align_monotonic_dp(cues, signs, DPConfig(),
                           boundary_start_probs=np.array([0.9, 5.]),
                           boundary_end_probs=np.array([0.9, 0.9]),
                           boundary_weight=1.0)


def test_weak_targets_do_not_label_nonoverlap():
    starts = np.array([0., 1., 2.], dtype=np.float32)
    ends = np.array([.9, 1.9, 2.9], dtype=np.float32)
    cues = [SubtitleCue(0.9, 1.0, "در شکاف"), SubtitleCue(0.1, 1.8, "واقعی")]
    labels, count = make_boundary_targets(starts, ends, cues)
    assert count == 1
    assert labels[0, 0] == 1.
    assert labels[1, 1] == 1.
    assert labels.sum() == 2.
