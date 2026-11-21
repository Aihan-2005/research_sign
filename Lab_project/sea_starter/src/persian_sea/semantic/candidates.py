from __future__ import annotations

"""Temporal candidate selection robust to overlapping E4s sign segments.

Start timestamps must be sorted, but midpoints may not be: a very long
segment can overlap later short segments. Never binary-search unsorted mids.
"""

from dataclasses import dataclass
import numpy as np


def select_candidate_indices(
    starts: np.ndarray, ends: np.ndarray,
    rough_start: float, rough_end: float, radius: float,
    min_candidates: int, max_candidates: int,
) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if starts.ndim != 1 or ends.ndim != 1 or len(starts) != len(ends) or not len(starts):
        raise ValueError("Invalid segment timestamp arrays")
    if (not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or np.any(np.diff(starts) < 0) or np.any(ends <= starts)):
        raise ValueError("Segments must be finite, start-sorted and have positive duration")
    if (not np.isfinite([rough_start, rough_end, radius]).all()
            or not (0 <= rough_start < rough_end) or radius < 0):
        raise ValueError("Invalid weak subtitle interval or search radius")
    if not (1 <= min_candidates <= max_candidates):
        raise ValueError("Need 1 <= min_candidates <= max_candidates")

    mids = (starts + ends) / 2.0
    cue_mid = (rough_start + rough_end) * 0.5
    lo_time, hi_time = rough_start - radius, rough_end + radius

    # Original version used np.searchsorted(mids), which assumes that mids is
    # sorted. That is not guaranteed when E4s produces overlapping intervals.
    selected = np.flatnonzero((mids >= lo_time) & (mids <= hi_time))
    if selected.size:
        lo, hi = int(selected.min()), int(selected.max()) + 1
    else:
        center = int(np.argmin(np.abs(mids - cue_mid)))
        lo, hi = center, center + 1

    n = len(mids)
    if hi - lo < min_candidates:
        center = int(np.argmin(np.abs(mids - cue_mid)))
        lo = max(0, center - min_candidates // 2)
        hi = min(n, lo + min_candidates)
        lo = max(0, hi - min_candidates)
    if hi - lo > max_candidates:
        center = int(np.argmin(np.abs(mids[lo:hi] - cue_mid))) + lo
        lo = max(0, center - max_candidates // 2)
        hi = min(n, lo + max_candidates)
        lo = max(0, hi - max_candidates)
    return np.arange(lo, hi, dtype=np.int64)


@dataclass(frozen=True)
class CandidateSpan:
    start_index: int
    end_index: int  # exclusive
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2.0


def enumerate_candidate_spans(
    starts: np.ndarray, ends: np.ndarray,
    max_span_segments: int, max_span_seconds: float,
) -> list[CandidateSpan]:
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if not (1 <= max_span_segments and max_span_seconds > 0):
        raise ValueError("Invalid span limits")
    spans = []
    for a in range(len(starts)):
        start = float(starts[a])
        max_end = start
        for b in range(a + 1, min(len(starts), a + max_span_segments) + 1):
            # An earlier E4s segment may extend beyond the next one's end.
            max_end = max(max_end, float(ends[b - 1]))
            if max_end <= start:
                continue
            if max_end - start > max_span_seconds:
                break
            spans.append(CandidateSpan(a, b, start, max_end))
    return spans


def interval_iou(start_a: float, end_a: float,
                 start_b: float, end_b: float) -> float:
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    if inter <= 0.0:
        return 0.0
    union = max(end_a, end_b) - min(start_a, start_b)
    return inter / union if union > 0 else 0.0


def temporal_prior(spans: list[CandidateSpan],
                   rough_start: float, rough_end: float,
                   sigma_seconds: float, duration_weight: float) -> np.ndarray:
    mid = (rough_start + rough_end) / 2.0
    duration = max(1e-3, rough_end - rough_start)
    sigma = max(1e-3, sigma_seconds)
    return np.asarray([
        -0.5 * ((s.midpoint - mid) / sigma) ** 2
        - duration_weight * abs(np.log(max(s.duration, 1e-3) / duration))
        for s in spans
    ], dtype=np.float32)
