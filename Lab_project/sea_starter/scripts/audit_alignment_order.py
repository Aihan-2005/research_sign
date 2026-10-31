from __future__ import annotations

import argparse
import json
from pathlib import Path

from persian_sea.config import load_config, read_video_ids
from persian_sea.io.subtitles import find_subtitle, pair_corrected_cues, read_subtitles


def audit_video(video_id: str, raw_dir: Path, gt_dir: Path) -> dict:
    raw_path = find_subtitle(raw_dir, video_id)
    gt_path = find_subtitle(gt_dir, video_id)
    if raw_path is None:
        return {"video_id": video_id, "status": "missing_raw_subtitles"}
    if gt_path is None:
        return {"video_id": video_id, "status": "missing_manual_alignment_gt"}

    raw = read_subtitles(raw_path)
    corrected = read_subtitles(gt_path)
    paired = pair_corrected_cues(raw, corrected)
    known = [(i, cue.mid) for i, cue in enumerate(paired) if cue is not None]

    comparisons = 0
    inversions = 0
    for left in range(len(known)):
        for right in range(left + 1, len(known)):
            comparisons += 1
            if known[left][1] > known[right][1]:
                inversions += 1

    rate = inversions / comparisons if comparisons else None
    return {
        "video_id": video_id,
        "status": "ok",
        "raw_cues": len(raw),
        "gt_cues": len(corrected),
        "paired_cues": len(known),
        "unpaired_or_unannotated_cues": len(raw) - len(known),
        "pairwise_comparisons": comparisons,
        "order_inversions": inversions,
        "inversion_rate": rate,
        "note": (
            "Unpaired cues are not proof of null alignment; they may be missing or "
            "ambiguous annotations, especially when subtitle text is duplicated."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure subtitle-order inversions against manually corrected timings."
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--ids", default="", help="Defaults to the validation video IDs")
    parser.add_argument("--output", default="outputs/reports/alignment_order_audit.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ids_path = Path(args.ids) if args.ids else cfg.paths.val_video_ids
    ids = read_video_ids(ids_path)
    rows = [audit_video(video_id, cfg.paths.subtitles_raw, cfg.paths.subtitles_gt)
            for video_id in ids]
    valid = [row for row in rows if row.get("status") == "ok"]
    total_pairs = sum(row["pairwise_comparisons"] for row in valid)
    total_inversions = sum(row["order_inversions"] for row in valid)
    report = {
        "ids_file": str(ids_path),
        "videos_requested": len(ids),
        "videos_with_manual_gt": len(valid),
        "pairwise_comparisons": total_pairs,
        "order_inversions": total_inversions,
        "global_inversion_rate": (
            total_inversions / total_pairs if total_pairs else None
        ),
        "videos": rows,
        "warning": (
            "This audit measures ordering only for cues that could be paired with "
            "manual GT. It does not prove that the full SAT/DTW decoder is beneficial."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(
        f"Videos with GT: {len(valid)}/{len(ids)}; "
        f"inversion rate: {report['global_inversion_rate']}"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()



