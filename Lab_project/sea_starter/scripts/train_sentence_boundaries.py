


from __future__ import annotations

import argparse
from dataclasses import replace

from persian_sea.translation.boundary import read_boundary_config, train_boundary_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sentence_boundaries.yaml")
    parser.add_argument("--max-train-videos", type=int, default=None,
                        help="Optional small smoke run; do not use this checkpoint for results")
    args = parser.parse_args()
    cfg = read_boundary_config(args.config)
    if args.max_train_videos is not None:
        cfg = replace(cfg, max_train_videos=args.max_train_videos).validate()
    checkpoint = train_boundary_detector(cfg)
    print(f"Best sentence-boundary checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
