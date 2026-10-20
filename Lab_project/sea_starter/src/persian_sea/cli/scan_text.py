import argparse

from persian_sea.config import load_config
from persian_sea.text.cache import scan_text_all


def main()-> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()

    scan_text_all(load_config(args.config))


if __name__ == '__main__':
    main()
    