from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from persian_sea.io.subtitles import read_subtitles


def percentile(values: list[float], p: float):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare proposed VTT timings; changes are not an accuracy score."
    )
    parser.add_argument("--ids", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--hybrid-dir", required=True)
    parser.add_argument("--output", default="outputs/reports/vid23_alignment_comparison.json")
    args = parser.parse_args()

    ids = [
        x.strip() for x in Path(args.ids).read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]
    base_dir, hybrid_dir = Path(args.baseline_dir), Path(args.hybrid_dir)
    rows = []
    for video_id in ids:
        base_path = base_dir / f"{video_id}.proposed.vtt"
        hybrid_path = hybrid_dir / f"{video_id}.proposed.vtt"
        if not base_path.is_file() or not hybrid_path.is_file():
            raise FileNotFoundError(f"Missing proposed VTT for {video_id}")
        base, hybrid = read_subtitles(base_path), read_subtitles(hybrid_path)
        n = min(len(base), len(hybrid))
        deltas = [
            max(abs(base[i].start - hybrid[i].start),
                abs(base[i].end - hybrid[i].end))
            for i in range(n)
        ]
        review_path = hybrid_dir / f"{video_id}.review.json"
        review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}
        actions = {}
        for decision in review.get("decisions", []):
            key = decision.get("action", "unknown")
            actions[key] = actions.get(key, 0) + 1
        rows.append({
            "video_id": video_id,
            "baseline_cues": len(base),
            "hybrid_cues": len(hybrid),
            "paired_cues": n,
            "cue_count_difference": len(hybrid) - len(base),
            "mean_change_seconds": statistics.mean(deltas) if deltas else None,
            "median_change_seconds": statistics.median(deltas) if deltas else None,
            "p90_change_seconds": percentile(deltas, 0.90),
            "max_change_seconds": max(deltas) if deltas else None,
            "changed_over_0_10s": sum(x > 0.10 for x in deltas),
            "changed_over_0_50s": sum(x > 0.50 for x in deltas),
            "changed_over_1_00s": sum(x > 1.00 for x in deltas),
            "quality_gate_actions": actions,
            "interpretation": "Change relative to SEA only; not ground-truth accuracy.",
        })

    result = {"baseline_dir": str(base_dir), "hybrid_dir": str(hybrid_dir), "videos": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
