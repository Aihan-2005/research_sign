from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from persian_sea.features.pose_reader import (
    RawHolisticPose,
    coordinates_to_pixels,
)


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def _square_box_from_points(
    points_xy: np.ndarray,
    width: int,
    height: int,
    padding: float,
    min_side: int = 32,
) -> Box | None:
    if points_xy.size == 0:
        return None

    points_xy = points_xy[np.isfinite(points_xy).all(axis=1)]
    if len(points_xy) < 2:
        return None

    xmin, ymin = points_xy.min(axis=0)
    xmax, ymax = points_xy.max(axis=0)
    cx = float((xmin + xmax) / 2.0)
    cy = float((ymin + ymax) / 2.0)

    side = max(float(xmax - xmin), float(ymax - ymin), float(min_side))
    side *= float(padding)

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2, y2)


def _points_for_component(component, frame_idx: int, width: int, height: int):
    xyz = component.xyz[frame_idx]
    conf = component.confidence[frame_idx]
    mask = conf > 0.05
    xy = xyz[mask, :2]
    return coordinates_to_pixels(xy, width, height)


def boxes_for_frame(
    pose: RawHolisticPose,
    frame_idx: int,
    width: int,
    height: int,
    hand_padding: float,
    face_padding: float,
    signer_padding: float,
) -> dict[str, Box | None]:
    body = _points_for_component(pose.body, frame_idx, width, height)
    left = _points_for_component(pose.left, frame_idx, width, height)
    right = _points_for_component(pose.right, frame_idx, width, height)

    face = np.empty((0, 2), dtype=np.float32)
    if pose.face is not None:
        face = _points_for_component(pose.face, frame_idx, width, height)

    signer_points = np.concatenate(
        [x for x in (body, left, right, face) if len(x)],
        axis=0,
    ) if any(len(x) for x in (body, left, right, face)) else np.empty((0, 2))

    return {
        "signer": _square_box_from_points(
            signer_points, width, height, signer_padding, min_side=96
        ),
        "left_hand": _square_box_from_points(
            left, width, height, hand_padding, min_side=40
        ),
        "right_hand": _square_box_from_points(
            right, width, height, hand_padding, min_side=40
        ),
        "face": _square_box_from_points(
            face, width, height, face_padding, min_side=64
        ),
    }


def crop_rgb(frame: np.ndarray, box: Box | None) -> np.ndarray | None:
    if box is None:
        return None
    crop = frame[box.y1:box.y2, box.x1:box.x2]
    if crop.size == 0:
        return None
    return crop
