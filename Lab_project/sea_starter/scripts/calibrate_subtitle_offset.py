from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(values: list[float]) -> dict:
    median = statistics.median(values)
    return {
        "count": len(values),
        "median_seconds": median,
        "mean_seconds": statistics.mean(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "mad_seconds": statistics.median(abs(x - median) for x in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate subtitle-to-sign offsets from manually checked landmarks. "
            "Positive offset means the sign occurs after the subtitle timestamp."
        )
    )
    parser.add_argument("--input", default="data/offset_landmarks.csv")
    parser.add_argument("--output", default="outputs/reports/offset_calibration.json")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(source)

    by_video: dict[str, list[float]] = defaultdict(list)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "subtitle_time", "sign_time"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for line_no, row in enumerate(reader, start=2):
            video_id = (row.get("video_id") or "").strip()
            if not video_id:
                continue
            try:
                subtitle_time = float(row["subtitle_time"])
                sign_time = float(row["sign_time"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid timestamp at CSV line {line_no}") from exc
            if not (math.isfinite(subtitle_time) and math.isfinite(sign_time)):
                raise ValueError(f"Non-finite timestamp at CSV line {line_no}")
            by_video[video_id].append(sign_time - subtitle_time)

    all_offsets = [value for values in by_video.values() for value in values]
    if not all_offsets:
        raise ValueError("No valid landmark rows were found")

    report = {
        "definition": "offset_seconds = sign_time - subtitle_time",
        "positive_means": "sign occurs after subtitle timestamp",
        "global": summarize(all_offsets),
        "videos": {video_id: summarize(values)
                   for video_id, values in sorted(by_video.items())},
        "suggested_yaml": {
            "subtitle_to_sign_offset_seconds": statistics.median(all_offsets),
            "subtitle_offsets_by_video": {
                video_id: statistics.median(values)
                for video_id, values in sorted(by_video.items())
            },
        },
        "warning": (
            "Use the suggested values only after checking landmark quality and "
            "dispersion. This script does not edit any training config automatically."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"Global median offset: {report['global']['median_seconds']:.3f}s")
    for video_id, row in report["videos"].items():
        print(
            f"{video_id}: median={row['median_seconds']:.3f}s "
            f"MAD={row['mad_seconds']:.3f}s n={row['count']}"
        )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
