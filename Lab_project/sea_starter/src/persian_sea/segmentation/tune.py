from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

from persian_sea.config import AppConfig, read_video_ids
from persian_sea.io.elan import read_sign_segments
from persian_sea.segmentation.legacy_e4s import LegacyE4sAdapter
from persian_sea.segmentation.metrics import SegmentationMetrics


@dataclass(frozen=True)
class CachedVideo:
    video_id: str
    inference: object
    gold: list


def _load_cache(cfg: AppConfig) -> tuple[LegacyE4sAdapter, list[CachedVideo]]:
    adapter = LegacyE4sAdapter(cfg.segmentation.model)
    cache: list[CachedVideo] = []

    for video_id in read_video_ids(cfg.paths.video_ids):
        pose = cfg.paths.poses / f"{video_id}.pose"
        gold_eaf = cfg.paths.segmentation_gt / f"{video_id}.eaf"

        if not pose.exists() or not gold_eaf.exists():
            print(f"[tune] skip {video_id}: missing pose or GT EAF")
            continue

        print(f"[tune] neural inference once: {video_id}")
        inference = adapter.infer(pose)
        gold, _ = read_sign_segments(gold_eaf, cfg.elan.gt_sign_tiers)
        cache.append(CachedVideo(video_id, inference, gold))

    if not cache:
        raise RuntimeError(
            f"No usable validation data under {cfg.paths.segmentation_gt}"
        )

    return adapter, cache


def _safe_for_sort(value: float) -> float:
    return value if math.isfinite(value) else 1e9


def tune_thresholds(cfg: AppConfig) -> dict:
    adapter, cache = _load_cache(cfg)
    primary = float(cfg.segmentation_eval.primary_iou_threshold)
    rows: list[dict] = []

    for b in cfg.segmentation_eval.tune_b_values:
        for o in cfg.segmentation_eval.tune_o_values:
            metrics = {
                float(t): SegmentationMetrics(float(t))
                for t in cfg.segmentation_eval.iou_thresholds
            }

            for item in cache:
                predicted = adapter.decode_signs(
                    item.inference,
                    b_threshold=int(b),
                    o_threshold=int(o),
                )
                for metric in metrics.values():
                    metric.update(predicted, item.gold)

            row = {
                "b_threshold": int(b),
                "o_threshold": int(o),
            }

            for threshold, metric in sorted(metrics.items()):
                prefix = f"iou_{threshold:.2f}"
                row[f"{prefix}_precision"] = metric.precision
                row[f"{prefix}_recall"] = metric.recall
                row[f"{prefix}_f1"] = metric.f1
                row[f"{prefix}_mean_matched_iou"] = metric.mean_matched_iou
                row[f"{prefix}_start_mae"] = metric.start_mae
                row[f"{prefix}_end_mae"] = metric.end_mae

            rows.append(row)
            print(
                f"[tune] B={b:>2} O={o:>2} "
                f"F1@{primary:.2f}={row[f'iou_{primary:.2f}_f1']:.4f}"
            )

    primary_f1 = f"iou_{primary:.2f}_f1"
    secondary_f1 = "iou_0.25_f1" if any(
        abs(float(x) - 0.25) < 1e-9
        for x in cfg.segmentation_eval.iou_thresholds
    ) else primary_f1

    best = max(
        rows,
        key=lambda row: (
            row[primary_f1],
            row[secondary_f1],
            -_safe_for_sort(row[f"iou_{primary:.2f}_start_mae"]),
            -_safe_for_sort(row[f"iou_{primary:.2f}_end_mae"]),
        ),
    )

    return {
        "num_validation_videos": len(cache),
        "primary_metric": primary_f1,
        "best": best,
        "rows": rows,
    }


def save_tuning_result(result: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "segmentation_threshold_search.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = result["rows"]
    if rows:
        with (report_dir / "segmentation_threshold_search.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    (report_dir / "best_segmentation_thresholds.json").write_text(
        json.dumps(result["best"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
