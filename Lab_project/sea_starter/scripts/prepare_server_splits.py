

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random
import tempfile

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm"}
SUBTITLE_EXTENSIONS = {".vtt", ".srt"}
SPLIT_NAMES = ("train", "val", "test")


def available_videos(videos_dir: Path, subtitles_dir: Path, *,
                     visual_cache_dir: Path | None = None) -> list[str]:
    if not videos_dir.is_dir() or not subtitles_dir.is_dir():
        raise FileNotFoundError("Both videos_dir and subtitles_dir must exist")
    videos = defaultdict(list)
    for path in videos_dir.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos[path.stem].append(path.name)
    ambiguous = {k: v for k, v in videos.items() if len(v) != 1}
    if ambiguous:
        raise ValueError(f"Duplicate video stems (different extensions): {ambiguous}")
    subtitles = {p.stem for p in subtitles_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in SUBTITLE_EXTENSIONS}
    ids = sorted(set(videos) & subtitles)
    if visual_cache_dir is not None:
        if not visual_cache_dir.is_dir():
            raise FileNotFoundError(f"Visual Cache directory does not exist: {visual_cache_dir}")
        ids = [vid for vid in ids if (visual_cache_dir / f"{vid}.npz").is_file()]
    if not ids:
        raise ValueError("No videos with matching subtitles (and optional caches) found")
    return ids


def read_metadata(path: Path, video_ids: list[str]) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        if "video_id" not in columns:
            raise ValueError("Metadata CSV must contain video_id column")
        rows = {}
        for row in reader:
            vid = (row.get("video_id") or "").strip()
            if not vid or vid in rows:
                raise ValueError(f"Blank/duplicate video_id in metadata: {vid!r}")
            rows[vid] = {key: (value or "").strip() for key, value in row.items()}
    missing = sorted(set(video_ids) - set(rows))
    if missing:
        raise ValueError(f"Metadata missing {len(missing)} selected videos: {missing[:8]}")
    return rows


def _group_by_source_and_signer(
    video_ids: list[str], metadata: dict[str, dict[str, str]], *,
    mode: str,
) -> dict[str, list[str]]:
    if mode == "video":
        return {vid: [vid] for vid in video_ids}
    if not metadata:
        raise ValueError(f"--group-by {mode} requires --metadata")
    key = "source_recording_id" if mode == "recording" else "signer_id"
    groups = defaultdict(list)
    for vid in video_ids:
        value = metadata[vid].get(key, "")
        if not value:
            raise ValueError(f"Missing {key} for {vid}; cannot safely group")
        groups[value].append(vid)
    return dict(groups)


def split_grouped(
    groups: dict[str, list[str]], *, seed: int = 42,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[str]]:
    """Reproducible greedy group assignment with safe minimum split sizes."""
    if len(groups) < 3:
        raise ValueError("At least 3 independent groups are required for train/val/test")
    if any(not (0 < f < 1) for f in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("Fractions must be positive and add to 1")
    n = sum(map(len, groups.values()))
    if n < 5:
        raise ValueError("Need at least 5 videos: train>=2, val>=2, test>=1")
    rng = random.Random(seed)
    entries = sorted(groups.items(), key=lambda x: x[0])
    rng.shuffle(entries)
    # Largest groups first, randomizing ties with the seeded shuffle.
    entries.sort(key=lambda x: -len(x[1]))
    out: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    targets = dict(zip(SPLIT_NAMES, (n * f for f in fractions)))

    # Seed each split with an entire independent group.
    for name, (_, ids) in zip(SPLIT_NAMES, entries[:3]):
        out[name].extend(ids)
        counts[name] += len(ids)
    for _, ids in entries[3:]:
        minima = {"train": 2, "val": 2, "test": 1}
        deficient = [name for name in SPLIT_NAMES if counts[name] < minima[name]]
        if deficient:
            winner = min(deficient, key=lambda name: (counts[name], SPLIT_NAMES.index(name)))
        else:
            # Assign to the split furthest below its desired proportion.
            winner = max(SPLIT_NAMES, key=lambda name: (
                (targets[name] - counts[name]) / max(1.0, targets[name]),
                -SPLIT_NAMES.index(name),
            ))
        out[winner].extend(ids)
        counts[winner] += len(ids)
    for name, minimum in (("train", 2), ("val", 2), ("test", 1)):
        if len(out[name]) < minimum:
            raise ValueError(
                f"Too few {name} videos ({len(out[name])} < {minimum}) with current groups; "
                "provide more independent groups or adjust group metadata"
            )
    for name in SPLIT_NAMES:
        out[name].sort()
    return out


def audit_group_leakage(
    splits: dict[str, list[str]], metadata: dict[str, dict[str, str]],
    *, strict_signer: bool,
) -> dict:
    id_sets = [set(splits[name]) for name in SPLIT_NAMES]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i+1, 3)):
        raise AssertionError("Video leaked across splits")
    results = {}
    for key in ("source_recording_id", "signer_id"):
        seen = defaultdict(set)
        missing = []
        for name in SPLIT_NAMES:
            for vid in splits[name]:
                value = metadata.get(vid, {}).get(key, "")
                if value:
                    seen[value].add(name)
                else:
                    missing.append(vid)
        collisions = {value: sorted(where) for value, where in seen.items()
                      if len(where) > 1}
        results[key] = {"cross_split_groups": collisions,
                        "missing_metadata_count": len(missing)}
    if metadata and results["source_recording_id"]["cross_split_groups"]:
        raise ValueError("Source recording leakage detected; use --group-by recording or signer")
    if strict_signer and (
        not metadata or results["signer_id"]["missing_metadata_count"] or
        results["signer_id"]["cross_split_groups"]
    ):
        raise ValueError("Signer-independent split not achieved; use --group-by signer and complete metadata")
    return results


def write_splits(output_dir: Path, splits: dict, report: dict, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"{name}_ids.txt" for name in SPLIT_NAMES}
    paths["all"] = output_dir / "all_ids.txt"
    paths["report"] = output_dir / "split_report.json"
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Split files already exist; refusing to change split silently. Use --overwrite.")
    # Commit all content after validation; no data files are copied or moved.
    for name in SPLIT_NAMES:
        target = paths[name]
        temp = target.with_name(target.name + ".tmp")
        temp.write_text("\n".join(splits[name]) + "\n", encoding="utf-8")
        os.replace(temp, target)
    all_ids = sorted(set().union(*(set(splits[name]) for name in SPLIT_NAMES)))
    temp_all = paths["all"].with_name("all_ids.txt.tmp")
    temp_all.write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    os.replace(temp_all, paths["all"])
    temp = paths["report"].with_name("split_report.json.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, paths["report"])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--subtitles-dir", type=Path, default=Path("data/subtitles_raw"))
    parser.add_argument("--visual-cache-dir", type=Path, default=None,
                        help="When supplied, only include videos with existing visual NPZ caches")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Optional CSV with video_id, source_recording_id, signer_id")
    parser.add_argument("--group-by", choices=("auto", "video", "recording", "signer"),
                        default="auto")
    parser.add_argument("--strict-signer", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/server_splits"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ids = available_videos(args.videos_dir, args.subtitles_dir,
                           visual_cache_dir=args.visual_cache_dir)
    metadata = read_metadata(args.metadata, ids) if args.metadata else {}
    group_by = args.group_by
    if group_by == "auto":
        group_by = "recording" if metadata else "video"
    groups = _group_by_source_and_signer(ids, metadata, mode=group_by)
    fractions = (args.train_frac, args.val_frac, args.test_frac)
    splits = split_grouped(groups, seed=args.seed, fractions=fractions)
    leakage = audit_group_leakage(splits, metadata, strict_signer=args.strict_signer)
    report = {
        "seed": args.seed, "group_by": group_by, "ratios_requested": list(fractions),
        "videos_dir": str(args.videos_dir), "subtitles_dir": str(args.subtitles_dir),
        "num_selected_videos": len(ids), "num_groups": len(groups),
        "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "ids": splits, "leakage_audit": leakage,
        "warnings": (
            ["Signer/source-recording independence UNVERIFIED (no metadata)."]
            if not metadata else
            (["Source recording IDs missing for some videos; source independence not fully verifiable."]
             if leakage["source_recording_id"]["missing_metadata_count"] else [])
            + (["Signer IDs missing for some videos; signer independence not fully verifiable."]
               if leakage["signer_id"]["missing_metadata_count"] else [])
            + (["Signer overlap exists; this is not signer-independent evaluation."]
               if leakage["signer_id"]["cross_split_groups"] else [])
        ),
    }
    write_splits(args.output_dir, splits, report, overwrite=args.overwrite)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": report["counts"],
                      "group_by": group_by, "warnings": report["warnings"]},
                     ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
