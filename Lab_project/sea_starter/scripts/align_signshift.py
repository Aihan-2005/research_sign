from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from persian_sea.alignment.pipeline import align_one
from persian_sea.alignment.signshift_prior import SignShiftBoundaryRuntime
from persian_sea.config import load_config, read_video_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SEA alignment with SignShift visual boundary probabilities."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--time-tolerance", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = replace(
        cfg,
        paths=replace(
            cfg.paths,
            video_ids=Path(args.video_ids),
            alignment_output=Path(args.output_dir),
        ),
        alignment=replace(
            cfg.alignment,
            overwrite=bool(args.overwrite),
        ),
    )
    runtime = SignShiftBoundaryRuntime(
        args.checkpoint,
        time_tolerance_seconds=args.time_tolerance,
    )
    for video_id in read_video_ids(cfg.paths.video_ids):
        align_one(
            video_id,
            cfg,
            signshift_runtime=runtime,
            backend_override="sea_signshift",
            boundary_weight=args.boundary_weight,
        )


if __name__ == "__main__":
    main()
