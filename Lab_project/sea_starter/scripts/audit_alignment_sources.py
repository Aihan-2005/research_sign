

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
from persian_sea.io.subtitles import find_subtitle, read_subtitles


def audit_one(video_id: str, raw_dir: str, settings: AlignmentSourceSettings) -> dict:
    path = find_subtitle(Path(raw_dir), video_id)
    if path is None:
        raise FileNotFoundError(f"Raw subtitle missing: {video_id}")
    raw = read_subtitles(path)
    selected, selected_path = load_alignment_cues(video_id, raw, settings)
    shifts = [max(abs(a.start-b.start), abs(a.end-b.end))
              for a,b in zip(raw, selected)]
    source_path = selected_path or path
    return {"video_id": video_id, "source": settings.mode,
            "num_cues": len(raw), "file": str(source_path),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "max_shift_seconds": max(shifts, default=0.),
            "mean_shift_seconds": sum(shifts)/len(shifts) if shifts else 0.,
            "cues_over_1_5s": sum(delta > 1.5 for delta in shifts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", required=True, help="UTF-8 video ID list")
    ap.add_argument("--raw-dir", default="data/subtitles_raw")
    ap.add_argument("--approved-dir", default="data/subtitles_approved")
    ap.add_argument("--sea-dir", default="outputs/aligned_subtitles")
    ap.add_argument("--semantic-dir", default="outputs/aligned_subtitles/semantic")
    ap.add_argument("--sources", nargs="+", default=["raw", "sea"],
                    choices=["raw", "sea", "semantic", "approved"])
    ap.add_argument("--report", default="outputs/reports/alignment_source_audit.json")
    args = ap.parse_args()
    video_ids = [x.strip() for x in Path(args.ids).read_text(encoding="utf-8").splitlines()
                 if x.strip() and not x.lstrip().startswith("#")]
    if not video_ids or len(video_ids) != len(set(video_ids)):
        ap.error("Empty/duplicate video IDs")
    results, errors = [], []
    for vid in video_ids:
        for mode in args.sources:
            try:
                row = audit_one(vid, args.raw_dir, AlignmentSourceSettings(
                    mode, args.approved_dir, args.sea_dir, args.semantic_dir))
                print(f"OK {vid} {mode}: {row['num_cues']} cues, "
                      f"mean shift={row['mean_shift_seconds']:.3f}s")
                results.append(row)
            except (OSError, ValueError) as exc:
                errors.append(f"{vid}/{mode}: {exc}")
                print("ERROR", errors[-1])
    report = {"ok": not errors, "results": results, "errors": errors,
              "warning": "Timestamp agreement is NOT sign-language translation accuracy."}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Report:", path)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
