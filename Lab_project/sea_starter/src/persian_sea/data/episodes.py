
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import shutil
import subprocess

from persian_sea.core.contracts import Episode
from persian_sea.io.subtitles import find_subtitle, read_subtitles
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")


@dataclass(frozen=True)
class EpisodeIssue:
    video_id: str
    severity: str
    code: str
    detail: str


def _find_video(root: Path, video_id: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        path = root / (video_id + ext)
        if path.is_file():
            return path
    return None


def duration_from_ffprobe(video: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ], capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0:
        return None
    try:
        value = float(result.stdout.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def load_episode(video_id: str, video_dir: Path, subtitle_dir: Path,
                 inspect_duration: bool = True) -> Episode:
    video_path = _find_video(Path(video_dir), video_id)
    subtitle_path = find_subtitle(Path(subtitle_dir), video_id)
    if video_path is None:
        raise FileNotFoundError(f"Video not found: {video_id}")
    if subtitle_path is None:
        raise FileNotFoundError(f"VTT/SRT not found: {video_id}")
    cues = tuple(read_subtitles(subtitle_path))
    if not cues:
        raise ValueError(f"No subtitles found: {subtitle_path}")
    duration = duration_from_ffprobe(video_path) if inspect_duration else None
    return Episode(video_id, video_path, subtitle_path, cues,
                   duration_seconds=duration)


def audit_episode(episode: Episode, max_overlap_seconds: float = 0.5,
                  duration_margin_seconds: float = 0.25) -> list[EpisodeIssue]:
    issues = []
    cues = episode.subtitles
    for i, cue in enumerate(cues):
        if not cue.text.strip():
            issues.append(EpisodeIssue(episode.video_id, "warning", "empty_text", f"cue {i}"))
        if episode.duration_seconds is not None and cue.end > episode.duration_seconds + duration_margin_seconds:
            issues.append(EpisodeIssue(episode.video_id, "error", "out_of_video", f"cue {i}: end {cue.end:.3f}s"))
        if i and cue.start < cues[i - 1].start:
            issues.append(EpisodeIssue(episode.video_id, "error", "out_of_order", f"cue {i}"))
        if i and cues[i - 1].end - cue.start > max_overlap_seconds:
            issues.append(EpisodeIssue(episode.video_id, "warning", "overlap", f"cue {i-1} and {i}"))
    return issues


def write_audit(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
