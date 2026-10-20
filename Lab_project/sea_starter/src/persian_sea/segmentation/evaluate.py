from pathlib import Path
import json

from persian_sea.config import AppConfig, read_video_ids, segmentation_subdir
from persian_sea.io.elan import read_sign_segments
from persian_sea.segmentation.metrics import BoundaryCounts, SegmentationMetrics


def evaluate_segmentation_directory(cfg: AppConfig) -> dict:
    prediction_dir = segmentation_subdir(cfg)

    iou_metrics = {
        float(t): SegmentationMetrics(float(t))
        for t in cfg.segmentation_eval.iou_thresholds
    }

    boundary_metrics: dict[tuple[str, int], BoundaryCounts] = {}
    for ms in cfg.segmentation_eval.boundary_tolerance_ms:
        boundary_metrics[("start", ms)] = BoundaryCounts()
        boundary_metrics[("end", ms)] = BoundaryCounts()

    used = 0
    skipped: list[str] = []

    for video_id in read_video_ids(cfg.paths.video_ids):
        pred_path = prediction_dir / f"{video_id}.eaf"
        gold_path = cfg.paths.segmentation_gt / f"{video_id}.eaf"

        if not pred_path.exists() or not gold_path.exists():
            skipped.append(video_id)
            continue

        predicted, _ = read_sign_segments(pred_path, cfg.elan.sign_tiers)
        gold, _ = read_sign_segments(gold_path, cfg.elan.gt_sign_tiers)

        for metric in iou_metrics.values():
            metric.update(predicted, gold)

        for ms in cfg.segmentation_eval.boundary_tolerance_ms:
            tolerance = ms / 1000.0
            boundary_metrics[("start", ms)].update(
                [x.start for x in predicted],
                [x.start for x in gold],
                tolerance,
            )
            boundary_metrics[("end", ms)].update(
                [x.end for x in predicted],
                [x.end for x in gold],
                tolerance,
            )

        used += 1

    return {
        "used_videos": used,
        "skipped_videos": skipped,
        "segment_detection": {
            f"iou_{t:.2f}": metric.as_dict()
            for t, metric in sorted(iou_metrics.items())
        },
        "boundary_detection": {
            f"{kind}_{ms}ms": metric.as_dict()
            for (kind, ms), metric in boundary_metrics.items()
        },
    }


def save_evaluation(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
