


from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from persian_sea.io.subtitles import find_subtitle, read_subtitles
from persian_sea.translation.data import VisualCacheLRU, load_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-ids", default="data/train_ids.txt")
    parser.add_argument("--val-ids", default="data/val_ids.txt")
    parser.add_argument("--test-ids", default="data/test_ids.txt")
    parser.add_argument("--visual-cache-dir", default="outputs/features")
    parser.add_argument("--subtitles-dir", default="data/subtitles_raw")
    parser.add_argument("--time-tolerance", type=float, default=2.0,
                        help="Allowed VTT overhang outside the clip, in seconds")
    parser.add_argument("--report", default="outputs/reports/boundary_data_audit.json")
    args = parser.parse_args()
    if args.time_tolerance < 0:
        parser.error("--time-tolerance must be >= 0")

    split_paths = {"train": args.train_ids, "val": args.val_ids, "test": args.test_ids}
    splits = {}
    for name, path in split_paths.items():
        if name == "test" and not Path(path).is_file():
            splits[name] = []
        else:
            splits[name] = load_ids(path)
    errors, warnings, rows = [], [], []
    sets = {name: set(ids) for name, ids in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(sets[a] & sets[b])
        if overlap:
            errors.append(f"Video IDs overlap between {a} and {b}: {overlap[:10]}")

    cache = VisualCacheLRU(1)
    for split, ids in splits.items():
        for video_id in ids:
            row = {"split": split, "video_id": video_id, "ok": False, "findings": [], "warnings": []}
            try:
                cache_path = Path(args.visual_cache_dir) / f"{video_id}.npz"
                visual = cache.get(cache_path)
                clip_start = float(visual["starts"][0])
                clip_end = float(visual["ends"][-1])
                duration = clip_end - clip_start
                subtitle_path = find_subtitle(Path(args.subtitles_dir), video_id)
                if subtitle_path is None:
                    raise FileNotFoundError(f"Subtitle not found for {video_id}")
                cues = read_subtitles(subtitle_path)
                if not cues:
                    raise ValueError("Subtitle file contains no cues")
                intersecting = [c for c in cues if c.end >= clip_start and c.start <= clip_end]
                out_of_range = [c for c in cues if c.start < clip_start - args.time_tolerance
                                or c.end > clip_end + args.time_tolerance]
                starts = np.asarray(visual["starts"])
                ends = np.asarray(visual["ends"])
                centers = (starts + ends) / 2.0
                estimated_lengths = []
                for cue in intersecting:
                    s_idx = int(np.argmin(np.abs(centers - np.clip(cue.start, clip_start, clip_end))) )
                    e_idx = int(np.argmin(np.abs(centers - np.clip(cue.end, clip_start, clip_end))) )
                    if e_idx >= s_idx:
                        estimated_lengths.append(e_idx - s_idx + 1)
                if np.any(np.diff(starts) < 0):
                    row["findings"].append("Segment starts are not ordered")
                if len(intersecting) == 0:
                    row["findings"].append(
                        "No VTT cue overlaps the visual cache; likely absolute-vs-clip-relative timestamp mismatch"
                    )
                outside_fraction = len(out_of_range) / max(1, len(cues))
                if out_of_range:
                    note = (f"{len(out_of_range)}/{len(cues)} cues extend beyond the first/last sign segment "
                            "by more than tolerance; this can be normal if no sign is present at clip edges")
                    row["warnings"].append(note)
                    warnings.append(f"{split}/{video_id}: {note}")
                if outside_fraction > 0.5:
                    row["findings"].append(
                        "More than half of VTT cues lie outside the visual segment time range; inspect clip-relative timestamps"
                    )
                if duration <= 0:
                    row["findings"].append("Invalid visual-cache duration")
                row.update({
                    "visual_cache": str(cache_path), "subtitle": str(subtitle_path),
                    "clip_start": clip_start, "clip_end": clip_end, "duration_seconds": duration,
                    "num_segments": len(starts), "num_cues": len(cues),
                    "cues_overlapping_clip": len(intersecting), "cues_outside_tolerance": len(out_of_range),
                    "outside_fraction": outside_fraction,
                    "estimated_segments_per_cue": {
                        "min": int(np.min(estimated_lengths)) if estimated_lengths else None,
                        "median": float(np.median(estimated_lengths)) if estimated_lengths else None,
                        "p90": float(np.percentile(estimated_lengths, 90)) if estimated_lengths else None,
                        "p95": float(np.percentile(estimated_lengths, 95)) if estimated_lengths else None,
                        "max": int(np.max(estimated_lengths)) if estimated_lengths else None,
                    },
                    "ok": len(intersecting) > 0 and outside_fraction <= 0.5 and duration > 0,
                })
                if not row["ok"]:
                    errors.append(f"{split}/{video_id}: " + "; ".join(row["findings"]))
            except (OSError, ValueError, KeyError, IndexError) as exc:
                row["findings"].append(str(exc))
                errors.append(f"{split}/{video_id}: {exc}")
            rows.append(row)
            if row["ok"]:
                print(f"OK {split}/{video_id}: {row['duration_seconds']:.1f}s, "
                      f"{row['num_segments']} segments, {row['num_cues']} cues")
                for note in row["warnings"]:
                    print(f"WARNING {split}/{video_id}: {note}")
            else:
                print(f"ERROR {split}/{video_id}: {'; '.join(row['findings'])}")

    report = {"ok": not errors, "split_counts": {k: len(v) for k, v in splits.items()},
              "checked_videos": len(rows), "errors": errors, "warnings": warnings,
              "videos": rows,
              "note": "VTT timing is approximate weak supervision, not human-verified boundary ground truth."}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
