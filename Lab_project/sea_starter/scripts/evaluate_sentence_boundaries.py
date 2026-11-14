

from __future__ import annotations

import argparse
import json

from persian_sea.translation.boundary import evaluate_boundary_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ids", default="data/test_ids.txt",
                        help="Must not contain training videos")
    parser.add_argument("--reference-dir", default="data/subtitles_gt",
                        help="Independently verified sentence boundaries in VTT/SRT")
    parser.add_argument("--output", default="outputs/reports/signshift_sentence_metrics.json")
    parser.add_argument("--allow-weak-reference", action="store_true",
                        help="Diagnostic only: opt in to approximate subtitle reference")
    args = parser.parse_args()
    report = evaluate_boundary_checkpoint(
        args.checkpoint,
        args.ids,
        args.reference_dir,
        allow_weak_reference=args.allow_weak_reference,
        output_file=args.output,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
