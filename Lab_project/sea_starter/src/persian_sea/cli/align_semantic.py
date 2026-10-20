import argparse
from pathlib import Path

from persian_sea.config import load_config, read_video_ids
from persian_sea.semantic.infer import align_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path; defaults to outputs/checkpoints/save_name",
    )
    parser.add_argument(
        "--video-ids",
        default=None,
        help="Optional id-list override; defaults to test_video_ids",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else cfg.paths.checkpoints / cfg.training.save_name
    )

    ids_path = (
        Path(args.video_ids)
        if args.video_ids
        else cfg.paths.test_video_ids
    )

    for video_id in read_video_ids(ids_path):
        align_video(
            video_id,
            cfg,
            checkpoint,
        )


if __name__ == "__main__":
    main()
