
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence

from persian_sea.core.contracts import AlignmentProposal, RefinementDecision, TimeSpan
from persian_sea.io.subtitles import SubtitleCue


@dataclass(frozen=True)
class QualityGateConfig:
    max_boundary_shift_seconds: float = 1.5
    auto_apply_small_shifts: bool = False

    def __post_init__(self):
        if self.max_boundary_shift_seconds < 0:
            raise ValueError("max_boundary_shift_seconds must be nonnegative")


def reconcile(
    originals: Sequence[SubtitleCue],
    proposals: Sequence[AlignmentProposal],
    config: QualityGateConfig,
) -> tuple[list[SubtitleCue], list[RefinementDecision]]:


    by_index = {}
    for x in proposals:
        if x.cue_index in by_index or x.cue_index < 0 or x.cue_index >= len(originals):
            raise ValueError("Duplicate or invalid proposal index")
        by_index[x.cue_index] = x
    adjusted = []
    decisions = []
    previous_start = -1.0
    for index, cue in enumerate(originals):
        original = TimeSpan(cue.start, cue.end)
        proposal = by_index.get(index)
        candidate = proposal.span if proposal is not None else None
        delta = max(abs(candidate.start - original.start),
                    abs(candidate.end - original.end)) if candidate else None
        if candidate is None:
            action = "no_proposal"
            applied = original
        elif candidate.start < previous_start:
            action = "review_order"
            applied = original
        elif delta > config.max_boundary_shift_seconds:
            action = "review_large_shift"
            applied = original
        elif config.auto_apply_small_shifts:
            action = "applied_small_shift" if delta > 0 else "unchanged"
            applied = candidate
        else:
            action = "suggested_only" if delta > 0 else "unchanged"
            applied = original
        if applied.start < previous_start:
            raise ValueError("Input subtitles/proposals are not monotonic in starts")
        previous_start = applied.start
        adjusted.append(SubtitleCue(applied.start, applied.end, cue.text))
        decisions.append(RefinementDecision(index, original, candidate, applied, action, delta))
    return adjusted, decisions
