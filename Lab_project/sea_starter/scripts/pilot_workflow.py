

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context(config_name: str, video_id: str):
    from persian_sea.config import load_config, segmentation_subdir
    from persian_sea.io.subtitles import find_subtitle
    from persian_sea.text.cache import text_cache_directory

    cfg = load_config(ROOT / config_name)
    video_paths = [cfg.paths.videos / (video_id + ext)
                   for ext in (".mp4", ".mov", ".mkv", ".webm")]
    found_videos = [p for p in video_paths if p.is_file()]
    if len(found_videos) != 1:
        raise FileNotFoundError(f"Expected exactly one video for {video_id}: {found_videos}")
    subtitles = find_subtitle(cfg.paths.subtitles_raw, video_id)
    if subtitles is None:
        raise FileNotFoundError(f"Subtitle VTT/SRT missing for {video_id}")
    return cfg, {
        "video": found_videos[0],
        "subtitle": subtitles,
        "pose": cfg.paths.poses / f"{video_id}.pose",
        "eaf": segmentation_subdir(cfg) / f"{video_id}.eaf",
        "text_npz": text_cache_directory(cfg) / f"{video_id}.npz",
        "text_json": text_cache_directory(cfg) / f"{video_id}.json",
        "visual_npz": cfg.paths.feature_cache / f"{video_id}.npz",
        "visual_json": cfg.paths.feature_cache / f"{video_id}.json",
    }


def check(args) -> None:
    from persian_sea.features.preflight import inspect_input
    from persian_sea.io.elan import read_sign_segments
    from persian_sea.text.cache import verify_text_cache
    from persian_sea.io.subtitles import read_subtitles

    cfg, paths = context(args.config, args.id)
    report = inspect_input(args.id, cfg.paths.videos, cfg.paths.subtitles_raw)
    errors = [entry for entry in report["findings"] if entry["severity"] == "error"]
    signs = []
    if paths["pose"].is_file():
        print(f"PASS  pose: {paths['pose']} ({paths['pose'].stat().st_size / 1024**2:.1f} MiB)")
    else:
        errors.append({"severity": "error", "code": "pose_missing", "detail": str(paths["pose"])})
    if paths["eaf"].is_file():
        try:
            signs, tier = read_sign_segments(paths["eaf"], cfg.elan.sign_tiers)
            print(f"PASS  E4s: {len(signs)} signs, tier={tier}")
            if not signs:
                errors.append({"severity": "error", "code": "empty_eaf", "detail": str(paths["eaf"])})
        except Exception as exc:
            errors.append({"severity": "error", "code": "eaf_read", "detail": str(exc)})
    else:
        errors.append({"severity": "error", "code": "eaf_missing", "detail": str(paths["eaf"])})

    if paths["text_npz"].is_file():
        try:
            verify_text_cache(paths["text_npz"], cfg, args.id)
            print(f"PASS  text cache: {paths['text_npz']}")
        except Exception as exc:
            errors.append({"severity": "error", "code": "text_cache", "detail": str(exc)})
    else:
        errors.append({"severity": "error", "code": "text_cache_missing", "detail": str(paths["text_npz"])})

    visual_exists = paths["visual_npz"].is_file() and paths["visual_json"].is_file()
    if visual_exists:
        print(f"PASS  visual cache present: {paths['visual_npz']}")
    elif args.require_visual:
        errors.append({"severity": "error", "code": "visual_missing", "detail": str(paths["visual_npz"])})
    else:
        print("INFO  visual cache not created yet: run DINOv2 on GPU")
    print(f"INFO  subtitles: {report['num_cues']} cues; duration={report['duration_seconds']} s")
    if report["findings"]:
        print("INFO  input findings:", json.dumps(report["findings"], ensure_ascii=False))
    for error in errors:
        print(f"ERROR {error['code']}: {error['detail']}")
    print(f"RESULT: {'PASS' if not errors else 'FAIL'} | {len(errors)} errors")
    if errors:
        raise SystemExit(2)


def align(args) -> None:
    from persian_sea.alignment.pipeline import align_one
    cfg, _ = context(args.config, args.id)
    out = align_one(args.id, cfg)
    proposal = out.with_name(f"{args.id}.proposed.vtt")
    review = out.with_name(f"{args.id}.review.json")
    print("SEA completed:")
    for path in (proposal, review, out):
        print(f" - {path} exists={path.exists()}")


def bundle(args) -> None:
    cfg, paths = context(args.config, args.id)
    keys = ("video", "subtitle", "pose", "eaf", "text_npz", "text_json")
    entries = [paths[k] for k in keys]
    missing = [str(p) for p in entries if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out in (p.resolve() for p in entries):
        raise ValueError("Do not overwrite an input file")
    id_file = Path("data/pilot_ids.txt")
    metadata = {
        "format": "persian-sea-single-video-v1",
        "video_id": args.id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    temp = out.with_suffix(out.suffix + ".part")
    if temp.exists():
        raise FileExistsError(f"Temporary destination exists: {temp}")
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as zf:
            for path in entries:
                relative = path.relative_to(ROOT) if path.is_absolute() else path
                zf.write(ROOT / relative, arcname=relative.as_posix())
                metadata["files"].append({
                    "path": relative.as_posix(),
                    "bytes": (ROOT / relative).stat().st_size,
                    "sha256": sha256_file(ROOT / relative),
                })
            zf.writestr("data/pilot_ids.txt", args.id + "\n")
            zf.writestr("pilot_manifest.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        os.replace(temp, out)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Bundle ready: {out} ({out.stat().st_size / 1024**2:.1f} MiB)")
    print("Transfer it to Google Drive once (or use VS Code's 'Upload to Colab').")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("check", "sea", "bundle"),
                    help="check=lightweight; sea=temporal baseline; bundle=Colab inputs")
    ap.add_argument("--id", default="news001", help="Video stem (no .mp4/.vtt)")
    ap.add_argument("--config", default="configs/pilot_mac.yaml")
    ap.add_argument("--require-visual", action="store_true", help="Fail until GPU visual cache arrives")
    ap.add_argument("--output", default="news001_inputs.zip", help="Destination for bundle action")
    args = ap.parse_args()
    os.chdir(ROOT)
    {"check": check, "sea": align, "bundle": bundle}[args.action](args)


if __name__ == "__main__":
    main()
