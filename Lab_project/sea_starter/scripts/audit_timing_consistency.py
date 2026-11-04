

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persian_sea.io.subtitles import read_subtitles, find_subtitle
from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
from persian_sea.alignment.timing_policy import expected_timing
from persian_sea.video_text.config import read_config as read_vlp
from persian_sea.translation.config import read_config as read_translation
from persian_sea.translation.boundary import read_boundary_config


def timing_for(video_id, raw, config, *, is_vlp: bool):
    aligned, _ = load_alignment_cues(video_id, raw, AlignmentSourceSettings(
        mode=config.pair_source,
        approved_dir=config.approved_subtitles_dir,
        sea_dir=config.sea_alignment_dir,
        semantic_dir=config.semantic_alignment_dir,
        max_shift_seconds=config.max_alignment_shift_seconds,
        require_alignment_review=config.require_alignment_review))
    return expected_timing(aligned, video_id, config.pair_source,
                           config.subtitle_to_sign_offset_seconds,
                           config.subtitle_offsets_by_video)


def compare(video_id, raw, config_by_stage):
    stage_cues = {name: timing_for(video_id, raw, cfg, is_vlp=(name == "vlp"))
                  for name, cfg in config_by_stage.items()}
    rows = []
    for idx, cue in enumerate(raw):
        stages = {name: {"start": cs[idx].start, "end": cs[idx].end}
                  for name, cs in stage_cues.items()}
        max_diff = max(max(abs(a["start"]-b["start"]), abs(a["end"]-b["end"]))
                       for a in stages.values() for b in stages.values())
        rows.append({"video_id": video_id, "cue_index": idx,
                     "text": cue.text, "clock_by_stage": stages,
                     "max_stage_difference_seconds": round(max_diff, 4),
                     "consistent": max_diff <= 0.001})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", default="data/val_ids.txt")
    p.add_argument("--vlp-config", default="configs/video_text_pretrain_first.yaml")
    p.add_argument("--translation-config", default="configs/translation_mt5_first.yaml")
    p.add_argument("--boundary-config", default="configs/sentence_boundaries.yaml")
    p.add_argument("--output", default="outputs/reports/timing_consistency.json")
    args = p.parse_args()
    configs = {"vlp": read_vlp(args.vlp_config),
               "translation": read_translation(args.translation_config),
               "boundary": read_boundary_config(args.boundary_config)}
    ids = [s.strip() for s in Path(args.ids).read_text(encoding="utf-8").splitlines()
           if s.strip() and not s.lstrip().startswith("#")]
    rows = []
    for video_id in ids:
        sub = find_subtitle(Path(configs["vlp"].subtitles_dir), video_id)
        if sub is None:
            raise FileNotFoundError(f"Raw subtitle missing: {video_id}")
        rows.extend(compare(video_id, read_subtitles(sub), configs))
    inconsistent = [r for r in rows if not r["consistent"]]
    result = {"num_videos": len(ids), "num_cues": len(rows),
              "inconsistent_cues": len(inconsistent),
              "examples": inconsistent[:50],
              "warning": "Timing agreement does not prove alignment correctness. "
                         "Evaluate with independent human GT."}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Timings: {len(rows)} cues, inconsistent={len(inconsistent)}; report={out}")
    if inconsistent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
