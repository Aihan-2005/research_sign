

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from persian_sea.io.subtitles import find_subtitle, read_subtitles
from persian_sea.alignment_sources import (
    AlignmentSourceSettings, REJECT_REVIEW_ACTIONS,
    load_alignment_cues, read_review_actions,
)
from persian_sea.alignment.timing_policy import resolve_offset, shift_interval


def sha256(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path else None


def build_rows(video_id: str, raw_dir: Path, settings: AlignmentSourceSettings,
               global_offset: float = 0.0, per_video: dict | None = None) -> list[dict]:
    source = find_subtitle(raw_dir, video_id)
    if source is None:
        raise FileNotFoundError(f"No raw subtitle for {video_id}")
    raw = read_subtitles(source)

    relaxed = AlignmentSourceSettings(
        settings.mode, settings.approved_dir, settings.sea_dir,
        settings.semantic_dir, settings.max_shift_seconds, False)
    selected, selected_path = load_alignment_cues(video_id, raw, relaxed)
    review_actions = None
    if settings.mode in {"sea", "semantic"}:
        review_actions = read_review_actions(selected_path, len(raw), required=False)
    offset = resolve_offset(video_id, settings.mode, global_offset, per_video)
    output = []
    for i, (a, b) in enumerate(zip(raw, selected)):
        s, e = shift_interval(b.start, b.end, offset)
        action = review_actions[i] if review_actions is not None else (
            "weak_raw" if settings.mode == "raw" else
            "approved_source" if settings.mode == "approved" else "missing_review")
        safe = action not in REJECT_REVIEW_ACTIONS and action != "missing_review"
        output.append({
            "video_id": video_id,
            "cue_index": i,
            "text": a.text,
            "raw_start": a.start, "raw_end": a.end,
            "matching_start": s, "matching_end": e,
            "max_boundary_shift_seconds": max(abs(a.start-s), abs(a.end-e)),
            "pair_source": settings.mode,
            "review_action": action,
            "weak_supervision": settings.mode != "approved",
            "valid_for_training": safe,
            "source_vtt": str(selected_path or source),
            "source_sha256": sha256(selected_path or source),
        })
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", default="data/train_ids.txt")
    p.add_argument("--raw-dir", default="data/subtitles_raw")
    p.add_argument("--pair-source", choices=sorted({"raw", "sea", "semantic", "approved"}), default="raw")
    p.add_argument("--approved-dir", default="data/subtitles_approved")
    p.add_argument("--sea-dir", default="outputs/aligned_subtitles")
    p.add_argument("--semantic-dir", default="outputs/aligned_subtitles/semantic")
    p.add_argument("--global-offset", type=float, default=0.0)
    p.add_argument("--offsets-json", default="", help="Optional JSON: {\"news001\":2.5,...}")
    p.add_argument("--output", default="outputs/reports/alignment_manifest.jsonl")
    args = p.parse_args()
    ids = [s.strip() for s in Path(args.ids).read_text(encoding="utf-8").splitlines()
           if s.strip() and not s.strip().startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("IDs list must be nonempty without duplicates")
    offsets = json.loads(Path(args.offsets_json).read_text(encoding="utf-8")) if args.offsets_json else {}
    if not isinstance(offsets, dict):
        raise ValueError("--offsets-json must be a mapping")
    settings = AlignmentSourceSettings(
        args.pair_source, args.approved_dir, args.sea_dir, args.semantic_dir)
    rows = [r for vid in ids for r in build_rows(
        vid, Path(args.raw_dir), settings, args.global_offset, offsets)]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                      encoding="utf-8")
    problems = [r for r in rows if not r["valid_for_training"]]
    report = {"videos": len(ids), "cues": len(rows),
              "unsafe_cues": len(problems), "unsafe_examples": problems[:30],
              "source": args.pair_source,
              "warning": "Proposed + suggested_only is weak, not verified ground truth. "
                         "Manually audit to support any accuracy claims."}
    target.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {target}; cues={len(rows)}, unsafe={len(problems)}")


if __name__ == "__main__":
    main()
