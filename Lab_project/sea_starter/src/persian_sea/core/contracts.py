
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable
import math


@dataclass(frozen=True)
class TimeSpan:
    start: float
    end: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end) and 0 <= self.start < self.end):
            raise ValueError(f"Invalid absolute seconds: {self.start}, {self.end}")


@dataclass(frozen=True)
class Episode:

    video_id: str
    video_path: Path
    subtitles_path: Path
    subtitles: tuple
    origin_seconds: float = 0.0
    duration_seconds: float | None = None

    def __post_init__(self):
        if not self.video_id or self.origin_seconds < 0:
            raise ValueError("Episode needs video_id and nonnegative origin_seconds")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")


@dataclass(frozen=True)
class AlignmentProposal:
    cue_index: int
    span: TimeSpan
    method: str
    score: float | None = None  



@dataclass(frozen=True)
class RefinementDecision:
    cue_index: int
    original: TimeSpan
    proposed: TimeSpan | None
    applied: TimeSpan
    action: str
    max_boundary_delta: float | None


@runtime_checkable
class EpisodeAligner(Protocol):
    def propose(self, subtitles: Sequence, signs: Sequence) -> list[AlignmentProposal]:

        ...
