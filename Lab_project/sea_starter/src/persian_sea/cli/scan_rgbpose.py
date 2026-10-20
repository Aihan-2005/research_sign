import argparse

from persian_sea.config import load_config
from persian_sea.features.scanner import scan_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan each predicted sign segment into cached Pose + DINOv2 RGB features."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    scan_all(load_config(args.config))


if __name__ == "__main__":
    main()
