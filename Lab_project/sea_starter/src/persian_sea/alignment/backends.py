from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from persian_sea.alignment.dp import DPConfig, align_monotonic_dp
from persian_sea.core.contracts import AlignmentProposal, TimeSpan
from persian_sea.core.registry import Registry


@dataclass
class IdentityAligner:
    def propose(self, subtitles: Sequence, signs: Sequence) -> list[AlignmentProposal]:
        return [
            AlignmentProposal(i, TimeSpan(c.start, c.end), "original")
            for i, c in enumerate(subtitles)
        ]


@dataclass
class SEATemporalAligner:
    dp_config: DPConfig

    def propose(self, subtitles: Sequence, signs: Sequence) -> list[AlignmentProposal]:
        if not signs:
            if self.dp_config.allow_null_alignment:
                return []
            raise ValueError("No sign units. Run/check segmentation or use original backend")
        raw = align_monotonic_dp(list(subtitles), list(signs), self.dp_config)
        if len(raw) != len(subtitles) and not self.dp_config.allow_null_alignment:
            raise ValueError("SEA DP did not align each subtitle; inspect segmentation")
        return [
            AlignmentProposal(x.cue_index, TimeSpan(x.start, x.end), "sea_temporal")
            for x in raw
        ]


alignment_backends = Registry("alignment backend")
alignment_backends.register("original", lambda **_: IdentityAligner())
alignment_backends.register("sea_temporal", lambda dp_config, **_: SEATemporalAligner(dp_config))
