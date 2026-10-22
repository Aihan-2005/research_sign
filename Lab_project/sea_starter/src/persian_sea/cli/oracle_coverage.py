
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from persian_sea.config import load_config
from persian_sea.diagnostics.oracle_coverage import CandidatePolicy, evaluate_oracle
from persian_sea.video_text.config import read_config as read_video_text_config


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".json",
                                 encoding="utf-8", delete=False) as fh:
        tmp = Path(fh.name)
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Oracle candidate recall on HUMAN-VERIFIED VTT only")
    p.add_argument("--config", required=True, help="Existing configs/baseline.yaml")
    p.add_argument("--video-text-config", help="Needed for --mode video_text/both")
    p.add_argument("--mode", choices=("semantic", "video_text", "both"), default="both")
    p.add_argument("--ids", required=True, help="Explicit validation/holdout ID file; NEVER train GT")
    p.add_argument("--report", default="outputs/reports/candidate_oracle.json")
    p.add_argument("--csv", help="Optional per-cue CSV path")
    p.add_argument("--require-complete-gt", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    requested = ("semantic", "video_text") if args.mode == "both" else (args.mode,)
    if "video_text" in requested and not args.video_text_config:
        p.error("--video-text-config is required for video_text/both")
    policies = {"semantic": CandidatePolicy.semantic(cfg)}
    if "video_text" in requested:
        vt = read_video_text_config(args.video_text_config)
        policies["video_text"] = CandidatePolicy.video_text(vt)
        if Path(vt.subtitles_dir) != cfg.paths.subtitles_raw or Path(vt.feature_cache_dir) != cfg.paths.feature_cache:
            p.error("Semantic and VideoText configs must use the same raw VTT and visual feature directories")
    output = {}
    for name in requested:
        output[name] = evaluate_oracle(
            args.ids, subtitle_dir=cfg.paths.subtitles_raw,
            verified_gt_dir=cfg.paths.subtitles_gt,
            feature_dir=cfg.paths.feature_cache, policy=policies[name],
            require_complete_gt=args.require_complete_gt,
        )
    _atomic_json(Path(args.report), output)
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [dict(mode=mode, **row) for mode, result in output.items()
                for row in result["per_cue"]]
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: value["summary"] for key, value in output.items()},
                     ensure_ascii=False, indent=2))
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
