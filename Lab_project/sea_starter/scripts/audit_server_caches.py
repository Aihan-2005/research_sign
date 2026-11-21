 
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate IDs in {path}")
    return ids


def audit_one(video_id: str, visual_path: Path, text_path: Path,
              *, expected_k: int, expected_views: list[str]) -> dict:
    problems = []
    info = {"video_id": video_id, "visual": str(visual_path),
            "text": str(text_path), "errors": problems}
    if not visual_path.is_file():
        problems.append("visual NPZ missing")
    else:
        try:
            with np.load(visual_path, allow_pickle=False) as f:
                required = {"pose", "rgb", "starts", "ends", "views"}
                missing = required.difference(f.files)
                if missing:
                    raise ValueError(f"Missing arrays: {sorted(missing)}")
                pose, rgb = f["pose"], f["rgb"]
                starts, ends = f["starts"], f["ends"]
                views = [str(v) for v in f["views"].tolist()]
                n = len(starts)
                if (not n or pose.ndim != 4 or rgb.ndim != 4
                        or pose.shape[:2] != (n, expected_k)
                        or pose.shape[-2:] != (75, 4)
                        or rgb.shape[:3] != (n, expected_k, len(expected_views))
                        or starts.shape != (n,) or ends.shape != (n,)
                        or views != expected_views
                        or np.any(np.diff(starts) < 0)
                        or np.any(ends <= starts)
                        or np.any(starts < 0)
                        or not all(np.isfinite(a).all() for a in (pose, rgb, starts, ends))):
                    raise ValueError("Invalid shape/timestamps/NaN or mismatched K/views")
                if "sample_times" in f:
                    sample_times = f["sample_times"]
                    if sample_times.shape != (n, expected_k):
                        raise ValueError("sample_times shape incompatible with K")
                info["segments"] = n
                info["pose_shape"] = list(pose.shape)
                info["rgb_shape"] = list(rgb.shape)
        except (OSError, ValueError, KeyError, EOFError) as exc:
            problems.append(f"invalid visual NPZ: {exc}")

    if not text_path.is_file():
        problems.append("text NPZ missing")
    else:
        try:
            with np.load(text_path, allow_pickle=False) as f:
                if not {"embeddings", "texts", "cache_identity_sha256"}.issubset(f.files):
                    raise ValueError("Missing embeddings/texts/cache_identity_sha256")
                vec, texts = f["embeddings"], f["texts"]
                if vec.ndim != 2 or vec.shape[0] != len(texts) or len(texts) == 0:
                    raise ValueError("Text embedding/cue count mismatch")
                if not np.isfinite(vec).all():
                    raise ValueError("Text embeddings have NaN/Inf")
                info["cues"] = len(texts)
                info["text_dimension"] = int(vec.shape[1])
            sidecar = text_path.with_suffix(".json")
            if not sidecar.is_file():
                raise ValueError("Text identity sidecar JSON is missing")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(meta.get("identity"), dict):
                raise ValueError("Text identity sidecar is not verifiable")
        except (OSError, ValueError, KeyError, EOFError) as exc:
            problems.append(f"invalid text cache: {exc}")
    return info


def audit(config_path: Path, split_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths, scanner, semantic = cfg["paths"], cfg["scanner"], cfg["semantic"]
    visual_root = Path(paths["feature_cache"])
    text_root = Path(paths["text_feature_cache"])
    if semantic.get("text_cache_layout") == "by_backend":
        text_root = text_root / semantic["text_backend"]
    ids_by_split = {name: read_ids(split_dir / f"{name}_ids.txt")
                    for name in ("train", "val", "test")}
    combined = [vid for ids in ids_by_split.values() for vid in ids]
    if len(combined) != len(set(combined)):
        raise ValueError("Video ID overlaps across train/val/test")
    results = [audit_one(vid, visual_root / f"{vid}.npz", text_root / f"{vid}.npz",
                         expected_k=int(scanner["samples_per_segment"]),
                         expected_views=list(scanner["views"]))
               for vid in sorted(combined)]
    errors = [f"{r['video_id']}: {item}" for r in results for item in r["errors"]]
    signatures = {
        (tuple(r.get("pose_shape", [])[1:]), tuple(r.get("rgb_shape", [])[1:]),
         r.get("text_dimension"))
        for r in results if not r["errors"]
    }
    if len(signatures) > 1:
        errors.append(f"Cross-video feature signatures mismatch: {signatures}")
    report = {
        "ok": not errors,
        "preprocess_config": str(config_path),
        "splits": {name: len(ids) for name, ids in ids_by_split.items()},
        "checked_videos": len(results), "errors": errors,
        "records": results,
        "note": "Structural cache audit only; separately use project text cache identity and VTT checks.",
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/server_pilot_preprocess.yaml"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/server_splits"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/server_pilot/cache_audit.json"))
    args = parser.parse_args(argv)
    report = audit(args.config, args.split_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Cache audit: {report['checked_videos']} videos, "
          f"{len(report['errors'])} errors. Report: {args.output}")
    for error in report["errors"][:20]:
        print("  ERROR:", error)
    if not report["ok"]:
        raise SystemExit(2)
    return report


if __name__ == "__main__":
    main()
