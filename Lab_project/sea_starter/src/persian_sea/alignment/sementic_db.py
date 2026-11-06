

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from persian_sea.io.subtitles import SubtitleCue
from persian_sea.types import Alignment, SignSegment


@dataclass(frozen=True)
class SemanticDPSettings:
    search_radius_seconds: float = 8.0
    max_group_segments: int = 56
    max_group_seconds: float = 22.0
    max_internal_gap_seconds: float = 6.0
    duration_weight: float = 1.0
    gap_weight: float = 2.0
    semantic_weight: float = 8.0
    skipped_sign_cost: float = 0.10
    allow_null_alignment: bool = True
    null_cue_cost: float = 14.0

    def validate(self) -> "SemanticDPSettings":
        values = [
            self.search_radius_seconds, self.max_group_seconds,
            self.max_internal_gap_seconds, self.duration_weight,
            self.gap_weight, self.semantic_weight,
            self.skipped_sign_cost, self.null_cue_cost,
        ]
        if not all(math.isfinite(float(x)) and x >= 0 for x in values):
            raise ValueError("DP weights, limits and costs must be nonnegative and finite")
        if self.max_group_segments < 1 or self.max_group_seconds <= 0:
            raise ValueError("Positive group length constraints required")
        if self.semantic_weight <= 0:
            raise ValueError("semantic_weight must be > 0 to enable semantic DP")
        return self


@dataclass(frozen=True)
class SemanticDPResult:
    proposals: tuple[Alignment, ...]
    unmatched_cue_indices: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    total_cost: float
    skipped_signs: int


def _validate_inputs(cues, signs, similarities):
    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.shape != (len(cues), len(signs)):
        raise ValueError(
            f"Similarity matrix shape {matrix.shape} != ({len(cues)},{len(signs)})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Similarity matrix contains NaN/Inf; do not align")
    if np.any(matrix < -1.0001) or np.any(matrix > 1.0001):
        raise ValueError("Expected normalized cosine similarity within [-1,1]")
    prev_start = -math.inf
    for cue in cues:
        if not (
            math.isfinite(cue.start) and math.isfinite(cue.end)
            and 0 <= cue.start < cue.end and cue.start >= prev_start
        ):
            raise ValueError("Subtitle cues must be finite, nonnegative, time-sorted")
        prev_start = cue.start
    prev_start = -math.inf
    for sign in signs:
        if not (
            math.isfinite(sign.start) and math.isfinite(sign.end)
            and 0 <= sign.start < sign.end and sign.start >= prev_start
        ):
            raise ValueError("Sign segments must be finite, nonnegative, time-sorted")
        prev_start = sign.start
    return matrix


def _spans_for_cue(
    cue: SubtitleCue,
    signs: Sequence[SignSegment],
    row: np.ndarray,
    settings: SemanticDPSettings,
):


    n = len(signs)
    if n == 0:
        return []
    mids = np.fromiter((s.mid for s in signs), dtype=np.float64, count=n)
    lo_time = cue.start - settings.search_radius_seconds
    hi_time = cue.end + settings.search_radius_seconds
    lo = int(np.searchsorted(mids, lo_time, side="left"))
    hi = int(np.searchsorted(mids, hi_time, side="right"))
    if lo >= hi:
        return []
    spans = []
    cumulative = np.concatenate(([0.0], np.cumsum(row, dtype=np.float64)))
    for a in range(lo, hi):
        total_gap = 0.0
        max_b = min(hi, a + settings.max_group_segments)
        for b in range(a + 1, max_b + 1):
            if b > a + 1:
                gap = max(0.0, signs[b - 1].start - signs[b - 2].end)
                if gap > settings.max_internal_gap_seconds:
                    break
                total_gap += gap
            start = signs[a].start
            end = signs[b - 1].end
            duration = end - start
            if duration <= 0:
                continue
            if duration > settings.max_group_seconds:
                break
            mean_sim = float((cumulative[b] - cumulative[a]) / (b - a))
            
            cost = (
                abs(cue.start - start)
                + abs(cue.end - end)
                + settings.duration_weight * abs(cue.duration - duration)
                + settings.gap_weight * total_gap
                - settings.semantic_weight * mean_sim
            )
            spans.append((a, b, float(cost), mean_sim))
    return spans


def align_semantic_dp(
    cues: Sequence[SubtitleCue],
    signs: Sequence[SignSegment],
    similarity_matrix: np.ndarray,
    settings: SemanticDPSettings,
) -> SemanticDPResult:

    settings.validate()
    cues, signs = list(cues), list(signs)
    sim = _validate_inputs(cues, signs, similarity_matrix)
    m, n = len(cues), len(signs)
    if not m:
        return SemanticDPResult((), (), (), n * settings.skipped_sign_cost, n)
    candidates = [
        _spans_for_cue(cue, signs, sim[i], settings)
        for i, cue in enumerate(cues)
    ]
    if not settings.allow_null_alignment and any(not row for row in candidates):
        raise ValueError("No feasible candidate for at least one cue (null disabled)")


    previous = np.arange(n + 1, dtype=np.float64) * settings.skipped_sign_cost
    back = []
    for i in range(m):
        current = np.full(n + 1, np.inf, dtype=np.float64)
        pointer = [None] * (n + 1)


        best_value = np.inf
        best_index = -1
        best_at = np.empty(n + 1, dtype=np.float64)
        best_source = np.empty(n + 1, dtype=np.int64)
        for a in range(n + 1):
            value = previous[a] - settings.skipped_sign_cost * a
            if value < best_value:
                best_value, best_index = value, a
            best_at[a] = best_value + settings.skipped_sign_cost * a
            best_source[a] = best_index

        if settings.allow_null_alignment:
            for j in range(n + 1):
                if np.isfinite(previous[j]):
                    current[j] = previous[j] + settings.null_cue_cost
                    pointer[j] = (j, None)

        for a, b, cost, _ in candidates[i]:
            if not np.isfinite(best_at[a]):
                continue
            new_cost = best_at[a] + cost
            if new_cost < current[b]:
                current[b] = new_cost
                pointer[b] = (int(best_source[a]), (a, b))

        if not np.isfinite(current).any():
            raise ValueError(
                f"No monotonic alignment path at cue {i}; widen search or enable null"
            )
        back.append(pointer)
        previous = current

    endings = previous + settings.skipped_sign_cost * (n - np.arange(n + 1))
    cursor = int(np.argmin(endings))
    total = float(endings[cursor])
    results = []
    nulls = []
    used = 0
    for i in reversed(range(m)):
        pointer = back[i][cursor]
        if pointer is None:
            raise RuntimeError("Missing DP backpointer")
        previous_cursor, selected = pointer
        if selected is None:
            nulls.append(i)
        else:
            a, b = selected
            used += b - a
            results.append(
                Alignment(
                    i, signs[a].start, signs[b - 1].end,
                    first_sign_index=a, last_sign_index=b - 1,
                )
            )
        cursor = previous_cursor

    results.reverse()
    nulls.reverse()
    return SemanticDPResult(
        tuple(results), tuple(nulls), tuple(map(len, candidates)),
        total, n - used,
    )
