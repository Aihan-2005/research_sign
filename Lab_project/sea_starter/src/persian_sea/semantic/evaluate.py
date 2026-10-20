from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from persian_sea.config import AppConfig, read_video_ids
from persian_sea.io.subtitles import find_subtitle, read_subtitles, pair_corrected_cues
from persian_sea.semantic.candidates import interval_iou


def evaluate_semantic_outputs(cfg: AppConfig, video_ids_path: Path) -> dict:


    rows = []
    missing_predictions = 0
    ignored_unverified = 0
    used_videos = 0
    for video_id in read_video_ids(video_ids_path):
        
        proposed = cfg.paths.alignment_output / "semantic" / f"{video_id}.proposed.vtt"
        applied = cfg.paths.alignment_output / "semantic" / f"{video_id}.vtt"
        pred_path = proposed if proposed.exists() else applied
        gt_path = find_subtitle(cfg.paths.subtitles_gt, video_id)
        if gt_path is None:
            ignored_unverified += 1
            continue
        if not pred_path.exists():
            missing_predictions += 1
            continue
        pred = read_subtitles(pred_path)
        gt = read_subtitles(gt_path)
        matched = pair_corrected_cues(pred, gt)
        used_videos += 1
        for cue, truth in zip(pred, matched):
            if truth is None:
                continue
            rows.append({
                "video_id": video_id, "text": cue.text,
                "iou": interval_iou(cue.start, cue.end, truth.start, truth.end),
                "start_error": abs(cue.start - truth.start),
                "end_error": abs(cue.end - truth.end),
            })
    if not rows:
        return {
            "num_verified_cues": 0, "used_videos": used_videos,
            "missing_prediction_videos": missing_predictions,
            "videos_without_verified_gt": ignored_unverified,
            "message": "No prediction/verified GT pairs found",
        }
    ious = np.asarray([x["iou"] for x in rows], dtype=np.float64)
    return {
        "used_videos": used_videos,
        "evaluated_variant": "proposed_if_available_else_applied",
        "num_verified_cues": len(rows),
        "missing_prediction_videos": missing_predictions,
        "videos_without_verified_gt": ignored_unverified,
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "accuracy_at_iou_0.10": float((ious >= 0.10).mean()),
        "accuracy_at_iou_0.25": float((ious >= 0.25).mean()),
        "accuracy_at_iou_0.50": float((ious >= 0.50).mean()),
        "start_mae_seconds": float(np.mean([x["start_error"] for x in rows])),
        "end_mae_seconds": float(np.mean([x["end_error"] for x in rows])),
    }


def save_semantic_evaluation(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
