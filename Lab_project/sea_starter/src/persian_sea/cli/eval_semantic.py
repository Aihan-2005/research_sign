import argparse

from persian_sea.config import load_config
from persian_sea.semantic.evaluate import (
    evaluate_semantic_outputs,
    save_semantic_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    result = evaluate_semantic_outputs(
        cfg,
        cfg.paths.test_video_ids,
    )

    output = (
        cfg.paths.reports
        / "semantic_alignment_evaluation.json"
    )
    save_semantic_evaluation(
        result,
        output,
    )

    print(result)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
