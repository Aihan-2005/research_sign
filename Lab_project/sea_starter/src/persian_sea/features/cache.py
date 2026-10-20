from __future__ import annotations

from pathlib import Path
import numpy as np


class FeatureCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=False)

    @property
    def pose(self) -> np.ndarray:
        return self.data["pose"]

    @property
    def rgb(self) -> np.ndarray:
        return self.data["rgb"]

    @property
    def starts(self) -> np.ndarray:
        return self.data["starts"]

    @property
    def ends(self) -> np.ndarray:
        return self.data["ends"]

    @property
    def views(self) -> list[str]:
        return [str(x) for x in self.data["views"].tolist()]

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "pose_shape": tuple(self.pose.shape),
            "pose_dtype": str(self.pose.dtype),
            "rgb_shape": tuple(self.rgb.shape),
            "rgb_dtype": str(self.rgb.dtype),
            "num_segments": int(len(self.starts)),
            "views": self.views,
            "rgb_hidden": int(self.data["rgb_hidden"][0]),
        }
