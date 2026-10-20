from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AlignmentCandidate:
    start: float
    end: float
    score: float
    local_rank: int


def monotonic_global_decode(
    candidates_per_cue: list[list[AlignmentCandidate]],
    rough_times: list[tuple[float, float]],
    overlap_penalty: float,
    gap_change_penalty: float,
) -> list[AlignmentCandidate]:


    if not candidates_per_cue:
        return []
    if len(candidates_per_cue) != len(rough_times):
        raise ValueError("Candidate/rough-time length mismatch")
    for i, group in enumerate(candidates_per_cue):
        if not group:
            raise ValueError(f"Cue {i} has no candidates")
        for c in group:
            if not (math.isfinite(c.score) and 0 <= c.start < c.end):
                raise ValueError(f"Invalid candidate for cue {i}: {c}")

    dp = [[float(c.score) for c in candidates_per_cue[0]]]
    back = [[-1] * len(candidates_per_cue[0])]

    for i in range(1, len(candidates_per_cue)):
        scores = [-math.inf] * len(candidates_per_cue[i])
        indices = [-1] * len(candidates_per_cue[i])
        rough_gap = rough_times[i][0] - rough_times[i - 1][1]
        for j, cur in enumerate(candidates_per_cue[i]):
            for k, prev in enumerate(candidates_per_cue[i - 1]):
                if not math.isfinite(dp[-1][k]):
                    continue
                if cur.start < prev.start:
                    continue
                overlap = max(0.0, prev.end - cur.start)
                actual_gap = cur.start - prev.end
                transition = (
                    - overlap_penalty * overlap
                    - gap_change_penalty * abs(actual_gap - rough_gap)
                )
                score = dp[-1][k] + float(cur.score) + transition
                if score > scores[j]:
                    scores[j], indices[j] = score, k
        if not any(math.isfinite(s) for s in scores):
            raise ValueError(
                f"No monotonic path at subtitle {i}. Increase global_top_k "
                "or search_radius_seconds; inspect candidate boundaries."
            )
        dp.append(scores)
        back.append(indices)

    last = max(range(len(dp[-1])), key=lambda j: dp[-1][j])
    path = [last]
    for i in range(len(candidates_per_cue) - 1, 0, -1):
        last = back[i][last]
        if last < 0:
            raise RuntimeError("Broken global decoder backtracking")
        path.append(last)
    path.reverse()
    return [candidates_per_cue[i][path[i]] for i in range(len(path))]
