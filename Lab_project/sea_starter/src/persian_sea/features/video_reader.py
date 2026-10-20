from __future__ import annotations

from pathlib import Path
import numpy as np
from decord import VideoReader, cpu


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")


def find_video(video_dir: Path, video_id: str) -> Path:
    for ext in VIDEO_EXTENSIONS:
        p = video_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No video for {video_id} in {video_dir}; tried {VIDEO_EXTENSIONS}"
    )


class IndexedVideoReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._vr = VideoReader(str(self.path), ctx=cpu(0), num_threads=2)
        self.length = len(self._vr)
        self.fps = float(self._vr.get_avg_fps())
        if self.length <= 0 or self.fps <= 0:
            raise RuntimeError(f"Invalid video metadata: {self.path}")

        first = self._vr[0].asnumpy()
        self.height, self.width = first.shape[:2]

    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        indices = np.clip(indices, 0, self.length - 1)
        # decord preserves requested order, including duplicates.
        return self._vr.get_batch(indices).asnumpy()
