
from __future__ import annotations
from pathlib import Path
import subprocess


from persian_sea.config import AppConfig, read_video_ids, segmentation_subdir
from persian_sea.pose.runner import find_video
from persian_sea.io.subtitles import find_subtitle


def build_segment_command(video_id: str, cfg: AppConfig) -> list[str]:
    pose_path = cfg.paths.poses / f"{video_id}.pose"
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    out_dir = segmentation_subdir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    eaf = out_dir / f"{video_id}.eaf"
    s = cfg.segmentation
    cmd = [
        "pose_to_segments", "--no-pose-link", f"--model={s.model}",
        f"--pose={pose_path}", f"--elan={eaf}",
        "--sign-b-threshold", str(s.sign_b_threshold),
        "--sign-o-threshold", str(s.sign_o_threshold),
    ]
    video = find_video(cfg.paths.videos, video_id)
    cmd.append(f"--video={video.resolve()}")
    subtitle = find_subtitle(cfg.paths.subtitles_raw, video_id)
    if subtitle is not None:
        cmd.append(f"--subtitles={subtitle.resolve()}")


    return cmd


def segment_one(video_id: str, cfg: AppConfig) -> Path:
    out_dir = segmentation_subdir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    eaf = out_dir / f"{video_id}.eaf"
    if eaf.exists() and not cfg.segmentation.overwrite:
        print(f"[segment] skip existing: {eaf}")
        return eaf
    cmd = build_segment_command(video_id, cfg)
    subprocess.run(cmd, check=True)
    if not eaf.exists() or eaf.stat().st_size == 0:
        raise RuntimeError(f"Expected EAF missing or empty: {eaf}")
    return eaf


def segment_all(cfg: AppConfig) -> None:
    for video_id in read_video_ids(cfg.paths.video_ids):
        segment_one(video_id, cfg)
