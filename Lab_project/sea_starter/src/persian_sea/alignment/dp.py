from __future__ import annotations

from dataclasses import dataclass
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


def _candidate_bounds(cues: list[Cue], signs: list[SignSegment],
                      window_size: int) -> list[tuple[int, int]]:
    mids = np.asarray([s.mid for s in signs], dtype=np.float64)
    bounds = []
    for cue in cues:
        nearest = np.argsort(np.abs(mids - cue.mid))[:min(window_size, len(signs))]
        bounds.append((int(nearest.min()), int(nearest.max()) + 1))
    return bounds


def _best_contiguous_subgroup(cue: Cue, group: list[SignSegment],
                              cfg: DPConfig) -> tuple[int, int]:
    if not group:
        return 0, 0
    chunks = []
    start = 0
    for i in range(1, len(group)):
        if group[i].start - group[i - 1].end > cfg.max_gap_seconds:
            chunks.append((start, i))
            start = i
    chunks.append((start, len(group)))
    return min(chunks, key=lambda ab: alignment_cost(
        cue, group[ab[0]:ab[1]], cfg.duration_penalty_weight,
        cfg.gap_penalty_weight,
    ))


def align_monotonic_dp(
    cues: list[Cue], signs: list[SignSegment], cfg: DPConfig,
    sim_matrix: np.ndarray | None = None,
) -> list[Alignment]:


    if not cues or not signs:
        return []
    m, n = len(cues), len(signs)
    if n < m:
        raise ValueError(f"Need signs>=cues: signs={n}, cues={m}")
    if sim_matrix is not None and sim_matrix.shape != (m, n):
        raise ValueError(f"sim_matrix must have shape {(m,n)}")
    if cfg.candidate_window_size < 1:
        raise ValueError("candidate_window_size must be >= 1")
    bounds = _candidate_bounds(cues, signs, cfg.candidate_window_size)
    dp = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    prev = np.full((m + 1, n + 1), -1, dtype=np.int64)

    dp[0, :] = 0.0

    prefix = None
    if sim_matrix is not None:
        prefix = np.zeros((m, n + 1), dtype=np.float64)
        prefix[:, 1:] = np.cumsum(sim_matrix, axis=1)

    for i in range(1, m + 1):
        cand_lo, cand_hi = bounds[i - 1]
        j_lo = max(i, cand_lo + 1)
        j_hi = min(n, max(j_lo, cand_hi))
        for j in range(j_lo, j_hi + 1):
            for k in range(i - 1, j):
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
                total = dp[i - 1, k] + cost
                if total < dp[i, j]:
                    dp[i, j], prev[i, j] = total, k

    best_j = int(np.argmin(dp[m]))
    if not np.isfinite(dp[m, best_j]):
        raise RuntimeError("SEA baseline DP found no valid path; inspect SIGN count/search window")
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
        a, b = _best_contiguous_subgroup(cue, group, cfg)
        selected = group[a:b]
        results.append(Alignment(
            cue_index=i,
            start=selected[0].start, end=selected[-1].end,
            first_sign_index=start_i + a, last_sign_index=start_i + b - 1,
        ))
    return results
