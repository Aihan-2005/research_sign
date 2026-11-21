from __future__ import annotations

"""Alignment backends; SignShift soft priors require explicit visual evidence."""

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from persian_sea.alignment.dp import DPConfig, align_monotonic_dp
from persian_sea.core.contracts import AlignmentProposal, TimeSpan
from persian_sea.core.registry import Registry


@dataclass
class IdentityAligner:
    def propose(self, subtitles: Sequence, signs: Sequence) -> list[AlignmentProposal]:
        return [AlignmentProposal(i, TimeSpan(c.start, c.end), "original")
                for i, c in enumerate(subtitles)]


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
        return [AlignmentProposal(x.cue_index, TimeSpan(x.start, x.end), "sea_temporal")
                for x in raw]


@dataclass
class SignShiftSEATemporalAligner:
    dp_config: DPConfig
    boundary_start_probs: np.ndarray
    boundary_end_probs: np.ndarray
    boundary_weight: float = 1.0

    def propose(self, subtitles: Sequence, signs: Sequence) -> list[AlignmentProposal]:
        if not signs:
            if self.dp_config.allow_null_alignment:
                return []
            raise ValueError("No sign units. Run/check segmentation first")
        raw = align_monotonic_dp(
            list(subtitles), list(signs), self.dp_config,
            boundary_start_probs=self.boundary_start_probs,
            boundary_end_probs=self.boundary_end_probs,
            boundary_weight=self.boundary_weight,
        )
        if len(raw) != len(subtitles) and not self.dp_config.allow_null_alignment:
            raise ValueError("SignShift+SEA DP did not align each cue")
        return [AlignmentProposal(x.cue_index, TimeSpan(x.start, x.end), "sea_signshift")
                for x in raw]


alignment_backends = Registry("alignment backend")
alignment_backends.register("original", lambda **_: IdentityAligner())
alignment_backends.register("sea_temporal", lambda dp_config, **_: SEATemporalAligner(dp_config))
alignment_backends.register(
    "sea_signshift",
    lambda dp_config, boundary_start_probs, boundary_end_probs,
           boundary_weight=1.0, **_: SignShiftSEATemporalAligner(
               dp_config, boundary_start_probs, boundary_end_probs, boundary_weight
           ),
)
