from __future__ import annotations

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
    if not np.all(np.diff(starts) >= 0) or np.any(ends <= starts):
        raise ValueError("Segments must be sorted and have positive duration")
    if not (0 <= rough_start < rough_end) or radius < 0:
        raise ValueError("Invalid weak subtitle interval or search radius")
    if not (1 <= min_candidates <= max_candidates):
        raise ValueError("Need 1 <= min_candidates <= max_candidates")

    mids = (starts + ends) / 2.0
    cue_mid = (rough_start + rough_end) * 0.5
    lo_time, hi_time = rough_start - radius, rough_end + radius
    lo = int(np.searchsorted(mids, lo_time, side="left"))
    hi = int(np.searchsorted(mids, hi_time, side="right"))
    lo = max(0, min(lo, len(mids) - 1))
    hi = max(lo + 1, min(hi, len(mids)))

    if hi - lo < min_candidates:
        center = int(np.argmin(np.abs(mids - cue_mid)))
        lo = max(0, center - min_candidates // 2)
        hi = min(len(mids), lo + min_candidates)
        lo = max(0, hi - min_candidates)
    if hi - lo > max_candidates:
        center = int(np.argmin(np.abs(mids[lo:hi] - cue_mid))) + lo
        lo = max(0, center - max_candidates // 2)
        hi = min(len(mids), lo + max_candidates)
        lo = max(0, hi - max_candidates)
    return np.arange(lo, hi, dtype=np.int64)


@dataclass(frozen=True)
class CandidateSpan:
    start_index: int
    end_index: int  
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
        for b in range(a + 1, min(len(starts), a + max_span_segments) + 1):
            start, end = float(starts[a]), float(ends[b - 1])
            if end <= start:
                continue
            if end - start > max_span_seconds:
                break
            spans.append(CandidateSpan(a, b, start, end))
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
