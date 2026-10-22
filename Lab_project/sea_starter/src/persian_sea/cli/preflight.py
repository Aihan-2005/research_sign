
import argparse
import json
from pathlib import Path
from persian_sea.config import load_config, read_video_ids
from persian_sea.features.preflight import inspect_input
from persian_sea.features.cache_audit import inspect_cache_pair
from persian_sea.text.cache import text_cache_directory, verify_text_cache


def main():
    ap = argparse.ArgumentParser(description="Audit video/VTT timebase and feature caches")
    ap.add_argument("--config", required=True)
    ap.add_argument("--video-ids", help="Optional subset of video IDs")
    ap.add_argument("--check-caches", action="store_true")
    ap.add_argument("--check-pose", action="store_true", help="Decode pose file and compare duration (uses RAM)")
    ap.add_argument("--report", help="Optional output JSON path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ids = read_video_ids(Path(args.video_ids) if args.video_ids else cfg.paths.video_ids)
    if not ids:
        ap.error("Empty list of video IDs")
    reports = []
    for vid in ids:
        item = inspect_input(vid, cfg.paths.videos, cfg.paths.subtitles_raw)
        if args.check_pose:
            from persian_sea.features.pose_quality import inspect_pose_timing
            if item.get("video"):
                item["pose_sync"] = inspect_pose_timing(cfg.paths.poses / f"{vid}.pose", Path(item["video"]))
        if args.check_caches:
            item["cache"] = inspect_cache_pair(vid, cfg.paths.feature_cache,
                                                 text_cache_directory(cfg),
                                                 cfg.paths.subtitles_raw,
                                                 expected_rgb_model=cfg.scanner.rgb_model,
                                                 expected_text_model=cfg.semantic.text_model,
                                                 expected_samples=cfg.scanner.samples_per_segment,
                                                 expected_views=cfg.scanner.resolved_views())

        
            cache_path = text_cache_directory(cfg) / f"{vid}.npz"
            if cache_path.exists():
                try:
                    verify_text_cache(cache_path, cfg, vid)
                except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
                    item["cache"]["findings"].append({
                        "severity": "error", "code": "text_cache_provenance",
                        "message": str(exc),
                    })
        reports.append(item)
    n_errors = sum(1 for row in reports for f in row["findings"] if f["severity"] == "error")
    n_errors += sum(1 for row in reports for f in row.get("cache", {}).get("findings", []) if f["severity"] == "error")
    n_errors += sum(1 for row in reports for f in row.get("pose_sync", {}).get("findings", []) if f["severity"] == "error")
    result = {"checked_videos": len(reports), "errors": n_errors, "videos": reports}
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if n_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
