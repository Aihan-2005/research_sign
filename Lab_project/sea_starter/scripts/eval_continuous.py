

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persian_sea.io.subtitles import find_subtitle, read_subtitles, normalize_persian_text
from persian_sea.translation.config import read_config as read_translation_config
from persian_sea.translation.continuous import ContinuousInferenceEngine
from persian_sea.translation.data import load_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-config", required=True)
    parser.add_argument("--translation-checkpoint", required=True)
    parser.add_argument("--boundary-checkpoint", default="outputs/sentence_boundaries/best_sentence_boundary.pt")
    parser.add_argument("--vlp-checkpoint", default=None)
    parser.add_argument("--ids", default="data/test_ids.txt")
    parser.add_argument("--reference-dir", default="data/subtitles_raw",
                        help="Read only after each video prediction has been generated")
    parser.add_argument("--output", default="outputs/translation/continuous_eval_test.json")
    args = parser.parse_args()
    try:
        import sacrebleu
    except ImportError as exc:
        raise SystemExit("Install evaluation dependency first: python -m pip install sacrebleu") from exc

    cfg = read_translation_config(args.translation_config)
    ids = load_ids(args.ids)
    engine = ContinuousInferenceEngine(
        cfg, args.translation_checkpoint, args.boundary_checkpoint,
        vlp_checkpoint=args.vlp_checkpoint,
    )
    predictions, references, rows, errors = [], [], [], []
    for index, video_id in enumerate(ids, start=1):
        try:

            result = engine.predict(video_id)

            subtitle = find_subtitle(Path(args.reference_dir), video_id)
            if subtitle is None:
                raise FileNotFoundError(f"Reference VTT/SRT missing: {video_id}")
            cues = read_subtitles(subtitle)
            reference = normalize_persian_text(" ".join(cue.text for cue in cues if cue.text.strip()))
            prediction = normalize_persian_text(result["transcript"].replace("\n", " "))
            predictions.append(prediction)
            references.append(reference)
            rows.append({"video_id": video_id, "num_predicted_sentences": result["num_predicted_sentences"],
                         "prediction": prediction, "reference": reference})
            print(f"[{index}/{len(ids)}] {video_id}: {result['num_predicted_sentences']} predicted sentences")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            errors.append({"video_id": video_id, "error": str(exc)})
            print(f"ERROR {video_id}: {exc}")
    if not predictions:
        raise SystemExit("No videos could be evaluated; inspect errors and paths")
    report = {
        "protocol": "continuous_visual_only_inference_reference_read_after_prediction",
        "num_requested_videos": len(ids), "num_scored_videos": len(predictions),
        "num_errors": len(errors),
        "corpus_bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
        "corpus_chrf": sacrebleu.corpus_chrf(predictions, [references]).score,
        "videos": rows, "errors": errors,
        "warning": "Boundary supervision is derived from approximate VTT. These scores are not human-verified ground-truth boundary metrics.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("protocol", "num_requested_videos", "num_scored_videos", "num_errors", "corpus_bleu", "corpus_chrf")}, ensure_ascii=False, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
