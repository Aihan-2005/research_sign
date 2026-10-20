from __future__ import annotations

import numpy as np


def sample_times(start: float, end: float, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be > 0")
    if end <= start:
        raise ValueError(f"invalid interval: {start} -> {end}")

    # Sample centers of equal temporal bins rather than interval endpoints.
    # This avoids over-emphasising transition frames exactly on boundaries.
    edges = np.linspace(start, end, count + 1, dtype=np.float64)
    return ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)


def times_to_indices(times: np.ndarray, fps: float, length: int) -> np.ndarray:
    if fps <= 0:
        raise ValueError(f"invalid fps: {fps}")
    if length <= 0:
        raise ValueError(f"invalid sequence length: {length}")

    idx = np.rint(times.astype(np.float64) * fps).astype(np.int64)
    return np.clip(idx, 0, length - 1)
