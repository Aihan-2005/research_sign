import argparse

from persian_sea.alignment.pipeline import align_all
from persian_sea.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    align_all(cfg)


if __name__ == "__main__":
    main()
