

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import shutil
import subprocess

from persian_sea.io.subtitles import find_subtitle, read_subtitles

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")


@dataclass(frozen=True)
class Finding:
    severity: str  # error / warning
    code: str
    detail: str


def video_probe(path: Path) -> dict | None:


    if not shutil.which("ffprobe"):
        return None
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration:stream=codec_type,duration,avg_frame_rate,r_frame_rate,width,height",
           "-of", "json", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
    if result.returncode:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[:300]}")
    return json.loads(result.stdout)


def inspect_input(video_id: str, videos: Path, subtitles: Path) -> dict:

    findings: list[Finding] = []
    matches = [videos / f"{video_id}{ext}" for ext in VIDEO_EXTENSIONS
               if (videos / f"{video_id}{ext}").is_file()]
    if len(matches) != 1:
        findings.append(Finding("error", "video_missing_or_ambiguous",
                                f"Found {len(matches)} videos for {video_id}"))
    subtitle = find_subtitle(subtitles, video_id)
    if subtitle is None:
        findings.append(Finding("error", "subtitle_missing", f"No VTT/SRT for {video_id}"))
    video_path = matches[0] if matches else None
    duration = None
    fps = None
    width = None
    height = None
    if video_path is not None:
        try:
            info = video_probe(video_path)
            if info is None:
                findings.append(Finding("warning", "ffprobe_unavailable", "Cannot verify video duration or dimensions"))
            else:
                duration = float(info["format"]["duration"])
                if not math.isfinite(duration) or duration <= 0:
                    findings.append(Finding("error", "invalid_duration", str(duration)))
                streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
                if not streams:
                    findings.append(Finding("error", "no_video_stream", str(video_path)))
                else:
                    width, height = streams[0].get("width"), streams[0].get("height")
                    fps = streams[0].get("avg_frame_rate")
                    if width is not None and height is not None and min(width, height) < 96:
                        findings.append(Finding("warning", "low_resolution", "Signer may be too small"))
                    if streams[0].get("avg_frame_rate") != streams[0].get("r_frame_rate"):
                        findings.append(Finding("warning", "timebase_check", "Avg/nominal frame rates differ; inspect timestamps (possible VFR)"))
        except (ValueError, KeyError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            findings.append(Finding("error", "video_probe_failed", str(exc)))
    num_cues = 0
    if subtitle is not None:
        try:
            cues = read_subtitles(subtitle)
            num_cues = len(cues)
            if not cues:
                findings.append(Finding("error", "no_cues", str(subtitle)))
            for i, cue in enumerate(cues):
                if not cue.text.strip():
                    findings.append(Finding("warning", "empty_cue", f"cue {i}"))
                if i and cue.start < cues[i-1].start:
                    findings.append(Finding("error", "out_of_order", f"cue {i}"))
                if duration is not None and cue.end > duration + 0.25:
                    findings.append(Finding("error", "outside_video", f"cue {i} ends at {cue.end:.3f}s > video {duration:.3f}s"))
                if i and cue.start < cues[i-1].end - 0.5:
                    findings.append(Finding("warning", "cue_overlap", f"cue {i-1}/{i}"))
        except (ValueError, OSError) as exc:
            findings.append(Finding("error", "subtitle_parse_failed", str(exc)))
    return {"video_id": video_id, "video": str(video_path) if video_path else None,
            "subtitle": str(subtitle) if subtitle else None,
            "duration_seconds": duration, "video_dimensions": [width, height],
            "frame_rate": fps, "num_cues": num_cues,
            "findings": [asdict(f) for f in findings]}
