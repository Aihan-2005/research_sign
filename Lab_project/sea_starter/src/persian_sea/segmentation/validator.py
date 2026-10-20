from dataclasses import asdict, dataclass
from pathlib import Path
import json
import statistics

from persian_sea.io.elan import read_sign_segments


@dataclass
class ValidationReport:
    file: str
    tier: str
    segment_count: int
    min_duration: float | None
    median_duration: float | None
    max_duration: float | None
    overlapping_pairs: int
    large_gaps_over_5s: int
    warnings: list[str]


def validate_eaf(
    path: str | Path,
    accepted_tiers: list[str],
) -> ValidationReport:
    path = Path(path)
    segments, tier = read_sign_segments(path, accepted_tiers)
    durations = [s.duration for s in segments]

    overlaps = 0
    large_gaps = 0
    warnings: list[str] = []

    for previous, current in zip(segments, segments[1:]):
        if current.start < previous.end:
            overlaps += 1
        if current.start - previous.end > 5.0:
            large_gaps += 1

    if not segments:
        warnings.append("No sign segments found.")
    if overlaps:
        warnings.append(f"{overlaps} overlapping consecutive pairs.")
    if durations and statistics.median(durations) < 0.08:
        warnings.append("Median sign duration < 80 ms; likely over-segmentation.")
    if durations and statistics.median(durations) > 3.0:
        warnings.append("Median sign duration > 3 s; likely merged signs.")

    return ValidationReport(
        file=str(path),
        tier=tier,
        segment_count=len(segments),
        min_duration=min(durations) if durations else None,
        median_duration=statistics.median(durations) if durations else None,
        max_duration=max(durations) if durations else None,
        overlapping_pairs=overlaps,
        large_gaps_over_5s=large_gaps,
        warnings=warnings,
    )


def save_validation_reports(
    reports: list[ValidationReport],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(x) for x in reports], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
