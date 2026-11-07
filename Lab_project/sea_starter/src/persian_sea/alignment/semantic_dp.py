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
        values = (
            self.search_radius_seconds,
            self.max_group_seconds,
            self.max_internal_gap_seconds,
            self.duration_weight,
            self.gap_weight,
            self.semantic_weight,
            self.skipped_sign_cost,
            self.null_cue_cost,
        )
        if not all(math.isfinite(float(value)) and value >= 0 for value in values):
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


def _validate_inputs(
    cues: Sequence[SubtitleCue],
    signs: Sequence[SignSegment],
    similarities: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(similarities, dtype=np.float64)
    expected_shape = (len(cues), len(signs))
    if matrix.shape != expected_shape:
        raise ValueError(f"Similarity matrix shape {matrix.shape} != {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Similarity matrix contains NaN/Inf; do not align")
    if np.any(matrix < -1.0001) or np.any(matrix > 1.0001):
        raise ValueError("Expected normalized cosine similarity within [-1,1]")

    previous_start = -math.inf
    for cue in cues:
        if not (
            math.isfinite(float(cue.start))
            and math.isfinite(float(cue.end))
            and 0 <= cue.start < cue.end
            and cue.start >= previous_start
        ):
            raise ValueError("Subtitle cues must be finite, nonnegative, time-sorted")
        previous_start = cue.start

    previous_start = -math.inf
    for sign in signs:
        if not (
            math.isfinite(float(sign.start))
            and math.isfinite(float(sign.end))
            and 0 <= sign.start < sign.end
            and sign.start >= previous_start
        ):
            raise ValueError("Sign segments must be finite, nonnegative, time-sorted")
        previous_start = sign.start

    return matrix


def _spans_for_cue(
    cue: SubtitleCue,
    signs: Sequence[SignSegment],
    row: np.ndarray,
    settings: SemanticDPSettings,
) -> list[tuple[int, int, float, float]]:


    count = len(signs)
    if count == 0:
        return []

    low_time = cue.start - settings.search_radius_seconds
    high_time = cue.end + settings.search_radius_seconds
    midpoints = np.fromiter((sign.mid for sign in signs), dtype=np.float64, count=count)
    cumulative_similarity = np.concatenate(
        ([0.0], np.cumsum(np.asarray(row, dtype=np.float64), dtype=np.float64))
    )

    spans: list[tuple[int, int, float, float]] = []
    for start_index in range(count):
        if not (low_time <= midpoints[start_index] <= high_time):
            continue

        total_gap = 0.0
        max_end = min(count, start_index + settings.max_group_segments)
        for end_exclusive in range(start_index + 1, max_end + 1):
            if end_exclusive > start_index + 1:
                gap = max(
                    0.0,
                    signs[end_exclusive - 1].start - signs[end_exclusive - 2].end,
                )
                if gap > settings.max_internal_gap_seconds:
                    break
                total_gap += gap

            last_midpoint = midpoints[end_exclusive - 1]
            if not (low_time <= last_midpoint <= high_time):
                continue

            start_time = float(signs[start_index].start)
            end_time = float(signs[end_exclusive - 1].end)
            duration = end_time - start_time
            if duration <= 0 or duration > settings.max_group_seconds:
                continue

            mean_similarity = float(
                (cumulative_similarity[end_exclusive]
                 - cumulative_similarity[start_index])
                / (end_exclusive - start_index)
            )
            cost = (
                abs(cue.start - start_time)
                + abs(cue.end - end_time)
                + settings.duration_weight * abs(cue.duration - duration)
                + settings.gap_weight * total_gap
                - settings.semantic_weight * mean_similarity
            )
            spans.append((start_index, end_exclusive, float(cost), mean_similarity))

    return spans


def align_semantic_dp(
    cues: Sequence[SubtitleCue],
    signs: Sequence[SignSegment],
    similarity_matrix: np.ndarray,
    settings: SemanticDPSettings,
) -> SemanticDPResult:


    settings.validate()
    cues = list(cues)
    signs = list(signs)
    similarities = _validate_inputs(cues, signs, similarity_matrix)

    cue_count, sign_count = len(cues), len(signs)
    if cue_count == 0:
        return SemanticDPResult(
            proposals=(),
            unmatched_cue_indices=(),
            candidate_counts=(),
            total_cost=sign_count * settings.skipped_sign_cost,
            skipped_signs=sign_count,
        )

    candidates = [
        _spans_for_cue(cue, signs, similarities[index], settings)
        for index, cue in enumerate(cues)
    ]
    if not settings.allow_null_alignment and any(not row for row in candidates):
        raise ValueError("No feasible candidate for at least one cue (null disabled)")


    previous = np.arange(sign_count + 1, dtype=np.float64) * settings.skipped_sign_cost
    backpointers: list[list[tuple[int, tuple[int, int] | None] | None]] = []

    for cue_index in range(cue_count):
        current = np.full(sign_count + 1, np.inf, dtype=np.float64)
        pointers: list[tuple[int, tuple[int, int] | None] | None] = [
            None
        ] * (sign_count + 1)


        best_value = np.inf
        best_source = -1
        best_at = np.full(sign_count + 1, np.inf, dtype=np.float64)
        best_source_at = np.full(sign_count + 1, -1, dtype=np.int64)
        for start_index in range(sign_count + 1):
            value = previous[start_index] - settings.skipped_sign_cost * start_index
            if value < best_value:
                best_value = value
                best_source = start_index
            best_at[start_index] = (
                best_value + settings.skipped_sign_cost * start_index
            )
            best_source_at[start_index] = best_source

        if settings.allow_null_alignment:
            for endpoint in range(sign_count + 1):
                if np.isfinite(previous[endpoint]):
                    current[endpoint] = previous[endpoint] + settings.null_cue_cost
                    pointers[endpoint] = (endpoint, None)

        for start_index, end_exclusive, span_cost, _ in candidates[cue_index]:
            source = int(best_source_at[start_index])
            if source < 0 or not np.isfinite(best_at[start_index]):
                continue
            new_cost = best_at[start_index] + span_cost
            if new_cost < current[end_exclusive]:
                current[end_exclusive] = new_cost
                pointers[end_exclusive] = (
                    source,
                    (start_index, end_exclusive),
                )

        if not np.isfinite(current).any():
            raise ValueError(
                f"No monotonic alignment path at cue {cue_index}; "
                "widen search or enable null"
            )
        backpointers.append(pointers)
        previous = current

    endpoints = np.arange(sign_count + 1)
    final_costs = previous + settings.skipped_sign_cost * (sign_count - endpoints)
    if not np.isfinite(final_costs).any():
        raise ValueError("No feasible final semantic alignment path")

    cursor = int(np.argmin(final_costs))
    total_cost = float(final_costs[cursor])
    proposals: list[Alignment] = []
    unmatched: list[int] = []
    used_signs = 0

    for cue_index in reversed(range(cue_count)):
        pointer = backpointers[cue_index][cursor]
        if pointer is None:
            raise RuntimeError("Missing DP backpointer during semantic alignment")
        previous_cursor, selected_span = pointer

        if selected_span is None:
            unmatched.append(cue_index)
        else:
            start_index, end_exclusive = selected_span
            used_signs += end_exclusive - start_index
            proposals.append(
                Alignment(
                    cue_index,
                    signs[start_index].start,
                    signs[end_exclusive - 1].end,
                    first_sign_index=start_index,
                    last_sign_index=end_exclusive - 1,
                )
            )
        cursor = previous_cursor

    proposals.reverse()
    unmatched.reverse()
    return SemanticDPResult(
        proposals=tuple(proposals),
        unmatched_cue_indices=tuple(unmatched),
        candidate_counts=tuple(len(row) for row in candidates),
        total_cost=total_cost,
        skipped_signs=sign_count - used_signs,
    )


__all__ = [
    "SemanticDPSettings",
    "SemanticDPResult",
    "align_semantic_dp",
]
