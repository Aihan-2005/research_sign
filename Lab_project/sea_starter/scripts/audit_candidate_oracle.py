
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from persian_sea.io.subtitles import read_subtitles, find_subtitle, pair_corrected_cues
from persian_sea.semantic.candidates import select_candidate_indices, interval_iou
from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
from persian_sea.alignment.timing_policy import expected_timing
from persian_sea.video_text.config import read_config


def examine_video(video_id: str, cfg, gt_dir: Path):
    raw_path = find_subtitle(Path(cfg.subtitles_dir), video_id)
    gt_path = find_subtitle(gt_dir, video_id)
    if raw_path is None:
        raise FileNotFoundError(f"Missing raw subtitle: {video_id}")
    if gt_path is None:
        return [], "missing_human_gt"
    raw = read_subtitles(raw_path)
    gt = pair_corrected_cues(raw, read_subtitles(gt_path))
    selected, _ = load_alignment_cues(video_id, raw, AlignmentSourceSettings(
        cfg.pair_source, cfg.approved_subtitles_dir, cfg.sea_alignment_dir,
        cfg.semantic_alignment_dir, cfg.max_alignment_shift_seconds,
        cfg.require_alignment_review))
    matched = expected_timing(selected, video_id, cfg.pair_source,
                              cfg.subtitle_to_sign_offset_seconds,
                              cfg.subtitle_offsets_by_video)
    path = Path(cfg.feature_cache_dir) / f"{video_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        starts, ends = data["starts"], data["ends"]
    result = []
    for i, (candidate_time, truth) in enumerate(zip(matched, gt)):
        if truth is None:
            continue
        idx = select_candidate_indices(
            starts, ends, candidate_time.start, candidate_time.end,
            radius=cfg.candidate_margin_seconds,
            min_candidates=cfg.min_candidate_segments,
            max_candidates=cfg.max_candidate_segments)
        choices = []
        for width in cfg.span_sizes:
            for j in range(0, max(0, len(idx)-width+1)):
                left, right = idx[j], idx[j+width-1]
                choices.append(interval_iou(
                    float(starts[left]), float(ends[right]), truth.start, truth.end))
        if not choices:
            raise ValueError(f"No valid VLP spans for {video_id}#{i}")
        oracle = float(max(choices))
        union = interval_iou(float(starts[idx[0]]), float(ends[idx[-1]]),
                             truth.start, truth.end)
        result.append({"video_id": video_id, "cue_index": i,
                       "subtitle_s": [raw[i].start, raw[i].end],
                       "matching_s": [candidate_time.start, candidate_time.end],
                       "human_gt_s": [truth.start, truth.end],
                       "candidate_count": len(idx),
                       "oracle_span_iou": oracle,
                       "candidate_envelope_iou": union,
                       "oracle_reaches_iou_0_5": oracle >= 0.5})
    return result, "ok"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vlp-config", default="configs/video_text_pretrain_first.yaml")
    p.add_argument("--ids", default="data/val_ids.txt")
    p.add_argument("--gt-dir", default="data/subtitles_gt")
    p.add_argument("--output", default="outputs/reports/candidate_oracle.json")
    args = p.parse_args()
    cfg = read_config(args.vlp_config)
    ids = [s.strip() for s in Path(args.ids).read_text(encoding="utf-8").splitlines()
           if s.strip() and not s.lstrip().startswith("#")]
    rows, status = [], {}
    for video_id in ids:
        result, stat = examine_video(video_id, cfg, Path(args.gt_dir))
        rows.extend(result)
        status[video_id] = stat
    summary = {
        "num_videos": len(ids), "num_human_gt_cues": len(rows),
        "oracle_iou_0_5_fraction": (
            sum(x["oracle_reaches_iou_0_5"] for x in rows)/len(rows) if rows else None),
        "videos": status, "rows": rows,
        "warning": "This is an upper-bound within the VLP span candidates, not learned alignment accuracy."
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Human GT cues={len(rows)}; oracle IoU>=0.5 fraction="
          f"{summary['oracle_iou_0_5_fraction']}; report={out}")
    if not rows:
        raise SystemExit("No matched human GT cues: cannot claim oracle accuracy")


if __name__ == "__main__":
    main()
