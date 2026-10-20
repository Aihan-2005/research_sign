from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


BODY = "POSE_LANDMARKS"
LEFT = "LEFT_HAND_LANDMARKS"
RIGHT = "RIGHT_HAND_LANDMARKS"
FACE = "FACE_LANDMARKS"


@dataclass
class ComponentArray:
    xyz: np.ndarray          # [T, V, 3]
    confidence: np.ndarray   # [T, V]


@dataclass
class RawHolisticPose:
    fps: float
    length: int
    body: ComponentArray
    left: ComponentArray
    right: ComponentArray
    face: ComponentArray | None


def _filled(x) -> np.ndarray:
    if hasattr(x, "filled"):
        x = x.filled(0)
    return np.asarray(x)


def _component(pose, name: str) -> ComponentArray:
    part = pose.get_components([name])
    data = _filled(part.body.data)
    conf = _filled(part.body.confidence)

    # Expected pose-format layout: [frames, people, points, xyz]
    if data.ndim != 4:
        raise ValueError(f"Unexpected {name} data shape: {data.shape}")
    if conf.ndim != 3:
        raise ValueError(f"Unexpected {name} confidence shape: {conf.shape}")

    xyz = data[:, 0, :, :3].astype(np.float32, copy=False)
    confidence = conf[:, 0, :].astype(np.float32, copy=False)
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
    return ComponentArray(xyz=xyz, confidence=confidence)


def load_holistic_pose(path: str | Path) -> RawHolisticPose:
    from pose_format import Pose

    path = Path(path)
    with path.open("rb") as f:
        pose = Pose.read(f.read())

    fps = float(pose.body.fps)
    length = int(pose.body.data.shape[0])

    body = _component(pose, BODY)
    left = _component(pose, LEFT)
    right = _component(pose, RIGHT)

    face = None
    try:
        face = _component(pose, FACE)
    except Exception:
        # Some pose extraction variants omit face landmarks. RGB face view then
        # falls back to the signer crop.
        face = None

    return RawHolisticPose(
        fps=fps,
        length=length,
        body=body,
        left=left,
        right=right,
        face=face,
    )


def _valid_points(xyz: np.ndarray, conf: np.ndarray, threshold: float = 0.05):
    valid = conf > threshold
    return xyz[valid], conf[valid]


def coordinates_to_pixels(
    xy: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Handle either normalized [0,1]-style or pixel-space coordinates."""
    xy = np.asarray(xy, dtype=np.float32).copy()
    if xy.size == 0:
        return xy

    finite = xy[np.isfinite(xy)]
    scale_like_normalized = finite.size and np.percentile(np.abs(finite), 95) <= 2.0

    if scale_like_normalized:
        xy[..., 0] *= width
        xy[..., 1] *= height

    return xy


def normalized_pose_tensor(
    pose: RawHolisticPose,
    frame_indices: np.ndarray,
    shoulder_epsilon: float = 0.02,
    include_confidence: bool = True,
) -> np.ndarray:
    """
    Return [K,75,4] (or [...,3]) using body+left+right.

    Coordinates are centred at the shoulder midpoint and divided by shoulder
    distance. This preserves relative hand/body geometry while reducing camera
    translation and scale effects.
    """
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    out = []

    for fi in frame_indices:
        body = pose.body.xyz[fi].copy()
        left = pose.left.xyz[fi].copy()
        right = pose.right.xyz[fi].copy()

        body_c = pose.body.confidence[fi].copy()
        left_c = pose.left.confidence[fi].copy()
        right_c = pose.right.confidence[fi].copy()

        # MediaPipe Pose landmark indices: left shoulder=11, right shoulder=12.
        if body.shape[0] <= 12:
            raise ValueError(f"Expected >=13 body landmarks, got {body.shape[0]}")

        center = (body[11, :3] + body[12, :3]) / 2.0
        shoulder = float(np.linalg.norm(body[11, :2] - body[12, :2]))
        shoulder = max(shoulder, shoulder_epsilon)

        xyz = np.concatenate([body, left, right], axis=0)
        conf = np.concatenate([body_c, left_c, right_c], axis=0)

        xyz = (xyz - center[None, :]) / shoulder
        xyz = np.clip(xyz, -8.0, 8.0)

        # Zero missing/very-low-confidence joints after normalization.
        missing = conf <= 0.01
        xyz[missing] = 0.0

        if include_confidence:
            feat = np.concatenate([xyz, conf[:, None]], axis=-1)
        else:
            feat = xyz

        out.append(feat.astype(np.float32))

    return np.stack(out, axis=0)
