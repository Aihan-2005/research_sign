import argparse

from persian_sea.config import load_config
from persian_sea.semantic.train import train_semantic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    checkpoint = train_semantic(
        load_config(args.config)
    )
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
