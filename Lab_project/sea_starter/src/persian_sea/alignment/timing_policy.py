

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from persian_sea.io.subtitles import SubtitleCue


def resolve_offset(
    video_id: str,
    pair_source: str,
    global_seconds: float = 0.0,
    per_video: Mapping[str, float] | None = None,
) -> float:
    """Use exactly one raw offset. Returns 0 for approved/SEA/semantic timings."""
    if pair_source not in {"raw", "sea", "semantic", "approved"}:
        raise ValueError(f"Unsupported timing source: {pair_source!r}")
    if pair_source != "raw":
        return 0.0
    value = float((per_video or {}).get(video_id, global_seconds))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite subtitle offset for {video_id}: {value!r}")
    return value


def shift_interval(start: float, end: float, offset: float) -> tuple[float, float]:

    if not all(map(math.isfinite, (float(start), float(end), float(offset)))):
        raise ValueError("Nonfinite interval or offset")
    if not (0.0 <= start < end):
        raise ValueError(f"Invalid original interval: ({start}, {end})")
    new_start = max(0.0, float(start) + float(offset))
    new_end = float(end) + float(offset)
    if new_end <= new_start:
        raise ValueError(
            f"Offset {offset:+.3f}s invalidates interval ({start}, {end}); "
            "review this cue or change the offset."
        )
    return new_start, new_end


def shift_cues(
    cues: Sequence[SubtitleCue], offset: float,
) -> list[SubtitleCue]:
    return [SubtitleCue(*shift_interval(c.start, c.end, offset), c.text)
            for c in cues]


def expected_timing(
    cues: Sequence[SubtitleCue], video_id: str,
    pair_source: str, global_seconds: float = 0.0,
    per_video: Mapping[str, float] | None = None,
) -> list[SubtitleCue]:
    return shift_cues(cues, resolve_offset(video_id, pair_source,
                                          global_seconds, per_video))
