
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path

from persian_sea.config import load_config, read_video_ids
from persian_sea.data.episodes import load_episode, audit_episode, write_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit video/subtitle file pairs and timings")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-ids", help="Optional split file")
    parser.add_argument("--out", help="Path for JSON audit")
    args = parser.parse_args()
    config = load_config(args.config)
    ids = read_video_ids(Path(args.video_ids) if args.video_ids else config.paths.video_ids)
    total_cues = 0
    results = []
    for video_id in ids:
        try:
            episode = load_episode(video_id, config.paths.videos, config.paths.subtitles_raw)
            issues = audit_episode(episode)
            total_cues += len(episode.subtitles)
            results.append({"video_id": video_id, "video": str(episode.video_path),
                            "subtitles": str(episode.subtitles_path),
                            "duration_seconds": episode.duration_seconds,
                            "num_cues": len(episode.subtitles),
                            "issues": [asdict(x) for x in issues]})
        except (OSError, ValueError) as exc:
            results.append({"video_id": video_id, "issues": [
                {"video_id": video_id, "severity": "error", "code": "load_failed", "detail": str(exc)}]})
    out = Path(args.out) if args.out else config.paths.reports / "dataset_audit.json"
    write_audit({"num_videos": len(ids), "num_cues": total_cues, "episodes": results}, out)
    print(f"Audit: {len(ids)} videos, {total_cues} cues → {out}")


if __name__ == "__main__":
    main()
