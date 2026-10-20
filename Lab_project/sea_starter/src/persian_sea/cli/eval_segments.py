import argparse
import json

from persian_sea.config import load_config
from persian_sea.segmentation.evaluate import (
    evaluate_segmentation_directory,
    save_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = evaluate_segmentation_directory(cfg)
    output = cfg.paths.reports / "segmentation_evaluation.json"
    save_evaluation(report, output)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
