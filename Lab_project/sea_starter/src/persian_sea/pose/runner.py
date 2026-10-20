from pathlib import Path
import subprocess

from persian_sea.config import AppConfig, read_video_ids


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")


def find_video(video_dir: Path, video_id: str) -> Path:
    for ext in VIDEO_EXTENSIONS:
        path = video_dir / f"{video_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Video {video_id} not found in {video_dir}. "
        f"Tried: {VIDEO_EXTENSIONS}"
    )


def _bool(value: bool) -> str:
    return "true" if value else "false"


def mediapipe_additional_config(cfg: AppConfig) -> str:
    p = cfg.pose
    return (
        f"model_complexity={p.model_complexity},"
        f"smooth_landmarks={_bool(p.smooth_landmarks)},"
        f"refine_face_landmarks={_bool(p.refine_face_landmarks)}"
    )


def build_pose_command(
    video_path: Path,
    output_path: Path,
    cfg: AppConfig,
) -> list[str]:
    return [
        "video_to_pose",
        "--format",
        cfg.pose.format,
        "-i",
        str(video_path),
        "-o",
        str(output_path),
        "--additional-config",
        mediapipe_additional_config(cfg),
    ]


def extract_pose_one(video_id: str, cfg: AppConfig) -> Path:
    video_path = find_video(cfg.paths.videos, video_id)
    output_path = cfg.paths.poses / f"{video_id}.pose"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not cfg.pose.overwrite:
        print(f"[pose] skip existing: {output_path}")
        return output_path

    cmd = build_pose_command(video_path, output_path, cfg)
    print("[pose]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Pose extraction failed: {output_path}")

    return output_path


def extract_pose_all(cfg: AppConfig) -> None:
    for video_id in read_video_ids(cfg.paths.video_ids):
        extract_pose_one(video_id, cfg)
