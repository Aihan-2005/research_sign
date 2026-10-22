from __future__ import annotations

import argparse
import json
from pathlib import Path

from persian_sea.config import load_config
from persian_sea.diagnostics.alignment_benchmark import VARIANTS, benchmark_alignments
from persian_sea.cli.oracle_coverage import _atomic_json


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate alignment variants against verified GT")
    p.add_argument("--config", required=True)
    p.add_argument("--ids", required=True, help="Validation/test source-video IDs")
    p.add_argument("--variants", nargs="+", choices=VARIANTS,
                   default=["raw", "sea_proposed", "semantic_proposed"])
    p.add_argument("--require-all", action="store_true")
    p.add_argument("--report", default="outputs/reports/alignment_suite.json")
    args = p.parse_args()
    result = benchmark_alignments(load_config(args.config), args.ids,
                                 tuple(args.variants), args.require_all)
    _atomic_json(Path(args.report), result)
    print(json.dumps({"common_verified_cues": result["common_verified_cues"],
                      "comparable": result["matched_common_method_metrics"],
                      "missing_prediction_videos": result["missing_prediction_videos"]},
                     ensure_ascii=False, indent=2))
    if result["common_verified_cues"] == 0:
        print("No common predictions yet. Generate requested methods first, or "
              "run --variants raw for the independent raw-time baseline.")


if __name__ == "__main__":
    main()
