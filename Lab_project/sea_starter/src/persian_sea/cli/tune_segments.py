import argparse
import json

from persian_sea.config import load_config
from persian_sea.segmentation.tune import (
    save_tuning_result,
    tune_thresholds,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = tune_thresholds(cfg)
    save_tuning_result(result, cfg.paths.reports)

    print("\nBEST THRESHOLDS")
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
