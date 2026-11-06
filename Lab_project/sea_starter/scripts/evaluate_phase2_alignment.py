"""Evaluate raw vs Phase-2 SEA semantic proposals on independently verified cues.

This script NEVER uses predicted/approved VTT as ground truth implicitly;
--gt-dir must point to manually verified human timing files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from persian_sea.io.subtitles import find_subtitle, read_subtitles, pair_corrected_cues
from persian_sea.semantic.candidates import interval_iou


def _summarize(rows: list[dict], prefix: str) -> dict:
    matched = [x for x in rows if x[f"{prefix}_available"]]
    if not matched:
        return {"n": 0, "start_mae_seconds": None, "end_mae_seconds": None,
                "mean_iou": None, "fraction_iou_at_least_0_5": None}
    return {
        "n": len(matched),
        "start_mae_seconds": float(np.mean([x[f"{prefix}_start_abs_error"] for x in matched])),
        "end_mae_seconds": float(np.mean([x[f"{prefix}_end_abs_error"] for x in matched])),
        "mean_iou": float(np.mean([x[f"{prefix}_iou"] for x in matched])),
        "fraction_iou_at_least_0_5": float(np.mean([x[f"{prefix}_iou"] >= 0.5 for x in matched])),
    }


def evaluate_one(video_id: str, raw_dir: Path, proposed_dir: Path, gt_dir: Path) -> dict:
    raw_path = find_subtitle(raw_dir, video_id)
    if raw_path is None:
        raise FileNotFoundError(f"{video_id}: raw subtitle missing")
    gt_path = find_subtitle(gt_dir, video_id)
    if gt_path is None:
        return {"video_id": video_id, "status": "missing_manual_gt", "rows": []}
    raw = read_subtitles(raw_path)
    verified = pair_corrected_cues(raw, read_subtitles(gt_path))
    proposal_path = proposed_dir / f"{video_id}.proposed.vtt"
    review_path = proposed_dir / f"{video_id}.review.json"
    proposal = read_subtitles(proposal_path) if proposal_path.is_file() else None
    if proposal is not None and ([c.text for c in proposal] != [c.text for c in raw]):
        raise ValueError(f"{video_id}: proposed cue count/order/text differs from raw")
    reviews = None
    status = "proposal_available" if proposal is not None else "proposal_missing_or_abstained"
    if review_path.is_file():
        report = json.loads(review_path.read_text(encoding="utf-8"))
        reviews = {int(d["cue_index"]): d.get("action") for d in report["decisions"]}
        if len(reviews) != len(raw):
            raise ValueError(f"{video_id}: corrupt/incomplete review action list")
    rows = []
    for i, (r, truth) in enumerate(zip(raw, verified)):
        if truth is None:
            continue
        row = {"video_id": video_id, "cue_index": i,
               "gt_start": truth.start, "gt_end": truth.end,
               "review_action": reviews.get(i) if reviews else "missing_review"}
        for prefix, item in (("raw", r), ("proposal", proposal[i] if proposal is not None else None)):
            row[f"{prefix}_available"] = item is not None
            if item is not None:
                row[f"{prefix}_start_abs_error"] = abs(item.start - truth.start)
                row[f"{prefix}_end_abs_error"] = abs(item.end - truth.end)
                row[f"{prefix}_iou"] = interval_iou(item.start, item.end, truth.start, truth.end)
        rows.append(row)
    return {"video_id": video_id, "status": status, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", default="data/val_ids.txt")
    p.add_argument("--raw-dir", default="data/subtitles_raw")
    p.add_argument("--proposed-dir", default="outputs/aligned_subtitles/sea_semantic_vlp")
    p.add_argument("--gt-dir", required=True, help="Manually verified GT; NOT model proposals")
    p.add_argument("--output", default="outputs/reports/phase2_semantic_alignment_eval.json")
    args = p.parse_args()
    ids = [x.strip() for x in Path(args.ids).read_text(encoding="utf-8").splitlines()
           if x.strip() and not x.lstrip().startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty/duplicate video IDs")
    values = [evaluate_one(vid, Path(args.raw_dir), Path(args.proposed_dir), Path(args.gt_dir))
              for vid in ids]
    rows = [row for result in values for row in result["rows"]]
    if not rows:
        raise ValueError("No matched human-verified cues: cannot report alignment accuracy")
    actions = Counter(x["review_action"] for x in rows)
    comparable = [r for r in rows if r["proposal_available"]]
    output = {
        "protocol": "independent_human_gt_sentence_level",
        "videos_requested": len(ids),
        "num_human_gt_cues": len(rows),
        "num_comparable_cues": len(comparable),
        "status_per_video": {r["video_id"]: r["status"] for r in values},
        "review_action_counts": dict(actions),
        "raw_all_human_gt": _summarize(rows, "raw"),
        "raw_same_comparable_cues": _summarize(comparable, "raw"),
        "semantic_proposals": _summarize(comparable, "proposal"),
        "rows": rows,
        "warning": ("Scored proposals are not certified GT. Compare raw/proposal only on SAME cues. "
                    "No semantic performance claims if manual ground truth is missing."),
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"GT cues={len(rows)}, comparable={len(comparable)}; report={dest}")
    print("RAW (comparable):", output["raw_same_comparable_cues"])
    print("SEA semantic:", output["semantic_proposals"])


if __name__ == "__main__":
    main()
