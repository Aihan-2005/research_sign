from __future__ import annotations

"""Monotonic SEA alignment, optionally guided by a soft visual boundary prior.

Default calls without boundary probabilities preserve temporal SEA behavior.
SignShift probabilities only add a bounded nonnegative penalty to the existing
cost; they are not semantic cosine similarities and are not human ground truth.
"""

from dataclasses import dataclass
import math
import numpy as np

from persian_sea.alignment.costs import alignment_cost
from persian_sea.types import Alignment, SignSegment
from persian_sea.io.subtitles import SubtitleCue as Cue


@dataclass
class DPConfig:
    duration_penalty_weight: float = 1.0
    gap_penalty_weight: float = 5.0
    max_gap_seconds: float = 10.0
    candidate_window_size: int = 50
    similarity_weight: float = 0.0
    allow_null_alignment: bool = False
    null_alignment_penalty: float = 30.0

    def __post_init__(self):
        if self.candidate_window_size < 1:
            raise ValueError("candidate_window_size must be >= 1")
        if not math.isfinite(float(self.null_alignment_penalty)) or self.null_alignment_penalty < 0:
            raise ValueError("null_alignment_penalty must be finite and nonnegative")


def _candidate_bounds(cues: list[Cue], signs: list[SignSegment],
                      window_size: int) -> list[tuple[int, int]]:
    mids = np.asarray([s.mid for s in signs], dtype=np.float64)
    bounds = []
    for cue in cues:
        nearest = np.argsort(np.abs(mids - cue.mid))[:min(window_size, len(signs))]
        bounds.append((int(nearest.min()), int(nearest.max()) + 1))
    return bounds


def _check_boundary_prior(
    starts, ends, weight: float, n: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("boundary_weight must be finite and nonnegative")
    if (starts is None) != (ends is None):
        raise ValueError("Both boundary probability arrays must be provided together")
    if starts is None:
        if weight:
            raise ValueError("Nonzero boundary_weight requires boundary probabilities")
        return None, None
    p = np.asarray(starts, dtype=np.float64).reshape(-1)
    q = np.asarray(ends, dtype=np.float64).reshape(-1)
    if p.shape != (n,) or q.shape != (n,):
        raise ValueError(f"Boundary probabilities must have shape ({n},)")
    if (not np.isfinite(p).all() or not np.isfinite(q).all()
            or np.any((p < 0) | (p > 1)) or np.any((q < 0) | (q > 1))):
        raise ValueError("Boundary probabilities must be finite and within [0,1]")
    return p, q


def _boundary_penalty(
    start_index: int, end_index: int,
    starts: np.ndarray | None, ends: np.ndarray | None, weight: float,
) -> float:
    if starts is None or weight == 0.0:
        return 0.0
    return float(weight * 0.5 * ((1.0 - starts[start_index]) + (1.0 - ends[end_index])))


def _best_contiguous_subgroup(
    cue: Cue, group: list[SignSegment], cfg: DPConfig, *,
    boundary_start_probs: np.ndarray | None = None,
    boundary_end_probs: np.ndarray | None = None,
    boundary_weight: float = 0.0,
    global_offset: int = 0,
) -> tuple[int, int]:
    if not group:
        return 0, 0
    chunks = []
    start = 0
    for i in range(1, len(group)):
        if group[i].start - group[i - 1].end > cfg.max_gap_seconds:
            chunks.append((start, i))
            start = i
    chunks.append((start, len(group)))

    def score(ab):
        a, b = ab
        return alignment_cost(
            cue, group[a:b], cfg.duration_penalty_weight,
            cfg.gap_penalty_weight,
        ) + _boundary_penalty(
            global_offset + a, global_offset + b - 1,
            boundary_start_probs, boundary_end_probs, boundary_weight,
        )
    return min(chunks, key=score)


def align_monotonic_dp(
    cues: list[Cue], signs: list[SignSegment], cfg: DPConfig,
    sim_matrix: np.ndarray | None = None, *,
    boundary_start_probs: np.ndarray | None = None,
    boundary_end_probs: np.ndarray | None = None,
    boundary_weight: float = 0.0,
) -> list[Alignment]:
    """Monotonic cue-to-sign DP, with optional null cues and SignShift prior."""
    n = len(signs)
    start_probs, end_probs = _check_boundary_prior(
        boundary_start_probs, boundary_end_probs, boundary_weight, n,
    )
    if not cues or not signs:
        return []

    m = len(cues)
    if n < m and not cfg.allow_null_alignment:
        raise ValueError(f"Need signs>=cues: signs={n}, cues={m}")
    if sim_matrix is not None and sim_matrix.shape != (m, n):
        raise ValueError(f"sim_matrix must have shape {(m, n)}")

    bounds = _candidate_bounds(cues, signs, cfg.candidate_window_size)
    dp = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    prev = np.full((m + 1, n + 1), -1, dtype=np.int64)
    dp[0, :] = 0.0  # Original SEA can skip leading unused signs.

    prefix = None
    if sim_matrix is not None:
        prefix = np.zeros((m, n + 1), dtype=np.float64)
        prefix[:, 1:] = np.cumsum(sim_matrix, axis=1)

    for i in range(1, m + 1):
        cand_lo, cand_hi = bounds[i - 1]

        if cfg.allow_null_alignment:
            for j in range(n + 1):
                if not np.isfinite(dp[i - 1, j]):
                    continue
                total = dp[i - 1, j] + cfg.null_alignment_penalty
                if total < dp[i, j]:
                    dp[i, j], prev[i, j] = total, j
            j_lo = max(1, cand_lo + 1)
            j_hi = min(n, max(j_lo, cand_hi))
            start_k = 0
        else:
            j_lo = max(i, cand_lo + 1)
            j_hi = min(n, max(j_lo, cand_hi))
            start_k = i - 1

        for j in range(j_lo, j_hi + 1):
            for k in range(start_k, j):
                if not np.isfinite(dp[i - 1, k]):
                    continue
                similarity_total = 0.0 if prefix is None else float(
                    prefix[i - 1, j] - prefix[i - 1, k]
                )
                cost = alignment_cost(
                    cues[i - 1], signs[k:j], cfg.duration_penalty_weight,
                    cfg.gap_penalty_weight, similarity_total=similarity_total,
                    similarity_weight=cfg.similarity_weight,
                )
                cost += _boundary_penalty(k, j - 1, start_probs, end_probs, boundary_weight)
                total = dp[i - 1, k] + cost
                if total < dp[i, j]:
                    dp[i, j], prev[i, j] = total, k

    best_j = int(np.argmin(dp[m]))
    if not np.isfinite(dp[m, best_j]):
        raise RuntimeError("SEA DP found no valid path; inspect SIGN count/search window")

    boundaries = [0] * (m + 1)
    boundaries[m] = best_j
    current = best_j
    for i in range(m, 0, -1):
        k = int(prev[i, current])
        if k < 0:
            raise RuntimeError("Broken DP backtracking")
        boundaries[i - 1] = k
        current = k

    results = []
    for i, cue in enumerate(cues):
        start_i, end_i = boundaries[i], boundaries[i + 1]
        group = signs[start_i:end_i]
        if not group:
            continue
        a, b = _best_contiguous_subgroup(
            cue, group, cfg,
            boundary_start_probs=start_probs,
            boundary_end_probs=end_probs,
            boundary_weight=boundary_weight,
            global_offset=start_i,
        )
        selected = group[a:b]
        results.append(Alignment(
            cue_index=i, start=selected[0].start, end=selected[-1].end,
            first_sign_index=start_i + a, last_sign_index=start_i + b - 1,
        ))
    return results
