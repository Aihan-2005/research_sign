import argparse

from persian_sea.config import load_config
from persian_sea.pose.runner import extract_pose_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose files for all project videos."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    extract_pose_all(load_config(args.config))


if __name__ == "__main__":
    main()
