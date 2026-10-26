
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from persian_sea.io.subtitles import SubtitleCue, find_subtitle, read_subtitles

SOURCES = frozenset({"raw", "approved", "sea", "semantic"})


@dataclass(frozen=True)
class AlignmentSourceSettings:
    mode: str = "raw"
    approved_dir: str = "data/subtitles_approved"
    sea_dir: str = "outputs/aligned_subtitles"
    semantic_dir: str = "outputs/aligned_subtitles/semantic"
    max_shift_seconds: float = 0.0  
    
    def validate(self) -> "AlignmentSourceSettings":
        if self.mode not in SOURCES:
            raise ValueError(f"Unknown pair_source {self.mode!r}; choose {sorted(SOURCES)}")
        if self.max_shift_seconds < 0:
            raise ValueError("max_shift_seconds cannot be negative")
        return self


def load_alignment_cues(video_id: str, raw_cues: Sequence[SubtitleCue],
                        settings: AlignmentSourceSettings) -> tuple[list[SubtitleCue], Path | None]:


    settings.validate()
    if not raw_cues:
        raise ValueError(f"{video_id}: empty raw subtitle list")
    if settings.mode == "raw":
        return list(raw_cues), None

    if settings.mode == "approved":
        source = find_subtitle(Path(settings.approved_dir), video_id)
    else:
        parent = Path(settings.sea_dir if settings.mode == "sea" else settings.semantic_dir)
        source = parent / f"{video_id}.proposed.vtt"
        if not source.is_file():
            source = None
    if source is None:
        raise FileNotFoundError(f"{video_id}: {settings.mode} timing file missing; no raw fallback")
    selected = read_subtitles(source)
    if len(selected) != len(raw_cues):
        raise ValueError(f"{video_id}: {settings.mode} cue count mismatch: "
                         f"{len(selected)} != {len(raw_cues)}")
    last_start = -1.0
    for i, (raw, aligned) in enumerate(zip(raw_cues, selected)):


        if raw.text != aligned.text:
            raise ValueError(f"{video_id}: {settings.mode} text/order changed at cue {i}")
        if aligned.start < last_start or aligned.start < 0 or aligned.end <= aligned.start:
            raise ValueError(f"{video_id}: invalid/non-monotonic timestamps at cue {i}")
        last_start = aligned.start
        delta = max(abs(raw.start - aligned.start), abs(raw.end - aligned.end))
        if settings.max_shift_seconds and delta > settings.max_shift_seconds:
            raise ValueError(f"{video_id}: {settings.mode} cue {i} shifted {delta:.3f}s "
                             f"> configured maximum {settings.max_shift_seconds:.3f}s")
    return selected, source
