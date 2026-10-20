import numpy as np

from persian_sea.semantic.candidates import (
    enumerate_candidate_spans,
    interval_iou,
    select_candidate_indices,
)


def test_candidate_selection_is_contiguous():
    starts = np.arange(20, dtype=np.float32)
    ends = starts + 0.6

    idx = select_candidate_indices(
        starts,
        ends,
        rough_start=8.0,
        rough_end=10.0,
        radius=2.0,
        min_candidates=4,
        max_candidates=6,
    )

    assert len(idx) <= 6
    assert np.all(np.diff(idx) == 1)


def test_span_enumeration():
    starts = np.array([0.0, 1.0, 2.0])
    ends = np.array([0.5, 1.5, 2.5])

    spans = enumerate_candidate_spans(
        starts,
        ends,
        max_span_segments=2,
        max_span_seconds=3.0,
    )

    pairs = [(s.start_index, s.end_index) for s in spans]
    assert (0, 1) in pairs
    assert (0, 2) in pairs
    assert (0, 3) not in pairs


def test_interval_iou():
    value = interval_iou(0.0, 2.0, 1.0, 3.0)
    assert abs(value - (1.0 / 3.0)) < 1e-8
