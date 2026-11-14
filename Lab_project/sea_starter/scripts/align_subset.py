from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from persian_sea.alignment.pipeline import align_all
from persian_sea.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SEA baseline on a selected ID list.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-ids", required=True)
    parser.add_argument("--output-dir", required=True)
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
        alignment=replace(cfg.alignment, overwrite=bool(args.overwrite), backend="sea_temporal"),
    )
    align_all(cfg)


if __name__ == "__main__":
    main()
