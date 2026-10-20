

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import math

from persian_sea.features.preflight import Finding, video_probe


def compare_pose_duration(*, pose_frames: int, pose_fps: float, video_seconds: float,
                          threshold_seconds: float = 0.5) -> dict:
    if pose_frames < 1 or not math.isfinite(pose_fps) or pose_fps <= 0:
        raise ValueError("Invalid pose frame count or fps")
    if not math.isfinite(video_seconds) or video_seconds <= 0:
        raise ValueError("Invalid video duration")
    pose_seconds = pose_frames / pose_fps
    delta = pose_seconds - video_seconds
    return {"pose_duration_seconds": pose_seconds,
            "video_duration_seconds": video_seconds,
            "duration_delta_seconds": delta,
            "within_tolerance": abs(delta) <= threshold_seconds}


def inspect_pose_timing(pose_path: Path, video_path: Path,
                        threshold_seconds: float = 0.5) -> dict:

    findings = []
    if not pose_path.is_file():
        findings.append(Finding("error", "pose_missing", str(pose_path)))
        return {"findings": [asdict(x) for x in findings]}
    try:
        from pose_format import Pose
        with pose_path.open("rb") as fp:
            pose = Pose.read(fp.read())
        frames = int(pose.body.data.shape[0])
        fps = float(pose.body.fps)
        meta = video_probe(video_path)
        if meta is None:
            findings.append(Finding("warning", "ffprobe_missing", "Cannot compare durations"))
            result = {"pose_frames": frames, "pose_fps": fps}
        else:
            streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]

            video_seconds = float(streams[0].get("duration") or meta["format"]["duration"])
            result = compare_pose_duration(pose_frames=frames, pose_fps=fps,
                                           video_seconds=video_seconds,
                                           threshold_seconds=threshold_seconds)
            result.update(pose_frames=frames, pose_fps=fps)
            if not result["within_tolerance"]:
                findings.append(Finding("error", "pose_video_duration_mismatch",
                    f"Pose and video differ by {result['duration_delta_seconds']:.3f}s; inspect frame timestamps"))
    except (ImportError, OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        findings.append(Finding("error", "pose_sync_check_failed", str(exc)))
        result = {}
    result["findings"] = [asdict(x) for x in findings]
    return result
