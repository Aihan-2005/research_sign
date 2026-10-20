import argparse

from persian_sea.config import load_config
from persian_sea.segmentation.runner import segment_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SEA-compatible E4s segmentation baseline."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    segment_all(load_config(args.config))


if __name__ == "__main__":
    main()
