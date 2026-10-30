


from __future__ import annotations

import argparse
from pathlib import Path

from persian_sea.translation.config import read_config as read_translation_config
from persian_sea.translation.continuous import ContinuousInferenceEngine, write_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-config", required=True)
    parser.add_argument("--translation-checkpoint", required=True)
    parser.add_argument("--boundary-checkpoint", default="outputs/sentence_boundaries/best_sentence_boundary.pt")
    parser.add_argument("--vlp-checkpoint", default=None,
                        help="Optional override if the VLP checkpoint moved; must be the same file content")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = read_translation_config(args.translation_config)
    engine = ContinuousInferenceEngine(
        cfg, args.translation_checkpoint, args.boundary_checkpoint,
        vlp_checkpoint=args.vlp_checkpoint,
    )
    result = engine.predict(args.video_id)
    default_output = Path(cfg.output_dir) / "continuous_predictions" / f"{args.video_id}.json"
    output = write_prediction(result, args.output or default_output)
    print(f"Predicted sentences: {result['num_predicted_sentences']}")
    print(f"Transcript JSON: {output}")
    if result["num_predicted_sentences"] == 0:
        print("WARNING: no boundary pairs passed the threshold. Inspect the boundary checkpoint/threshold and validation report.")
    print(result["transcript"])


if __name__ == "__main__":
    main()
