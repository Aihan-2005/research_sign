
from persian_sea.io.subtitles import SubtitleCue as Cue
from persian_sea.types import SignSegment


def total_internal_gap(signs: list[SignSegment]) -> float:
    if len(signs) < 2:
        return 0.0
    return sum(
        max(0.0, signs[i].start - signs[i - 1].end)
        for i in range(1, len(signs))
    )


def alignment_cost(
    cue: Cue,
    signs: list[SignSegment],
    duration_weight: float,
    gap_weight: float,
    similarity_total: float = 0.0,
    similarity_weight: float = 0.0,
) -> float:
    if not signs:
        return float("inf")

    group_start = signs[0].start
    group_end = signs[-1].end
    group_duration = group_end - group_start

    return (
        abs(cue.start - group_start)
        + abs(cue.end - group_end)
        + duration_weight * abs(cue.duration - group_duration)
        + gap_weight * total_internal_gap(signs)
        - similarity_weight * similarity_total
    )
