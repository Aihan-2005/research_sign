
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


def choose_device(requested: str = "auto") -> str:
    """Local-only device selection; main project training config is unchanged."""
    if requested != "auto":
        if requested not in {"cpu", "mps", "cuda"}:
            raise ValueError("--device must be auto, cpu, mps, or cuda")
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _inspect(cfg, vid: str) -> dict:
    from persian_sea.config import segmentation_subdir
    from persian_sea.features.preflight import inspect_input
    from persian_sea.io.elan import read_sign_segments
    from persian_sea.io.subtitles import find_subtitle, read_subtitles
    from persian_sea.text.cache import text_cache_directory, verify_text_cache

    result = {"video_id": vid, "mode": "single_video_smoke_no_training",
              "errors": [], "warnings": [], "heavy_model_loaded": False}
    input_result = inspect_input(vid, cfg.paths.videos, cfg.paths.subtitles_raw)
    result["input"] = input_result
    for issue in input_result["findings"]:
        result["errors" if issue["severity"] == "error" else "warnings"].append(
            f"input/{issue['code']}: {issue['detail']}")
    if result["errors"]:
        return result

    source_vtt = find_subtitle(cfg.paths.subtitles_raw, vid)
    result["subtitle_cues"] = len(read_subtitles(source_vtt))

    pose_path = cfg.paths.poses / f"{vid}.pose"
    result["pose_path"] = str(pose_path)
    if not pose_path.is_file() or pose_path.stat().st_size == 0:
        result["errors"].append(f"Missing or empty pose: {pose_path}")
    else:
        result["pose_bytes"] = pose_path.stat().st_size

    eaf_path = segmentation_subdir(cfg) / f"{vid}.eaf"
    result["eaf_path"] = str(eaf_path)
    if not eaf_path.is_file():
        result["errors"].append(f"Missing EAF: {eaf_path}")
    else:
        try:
            signs, tier = read_sign_segments(eaf_path, cfg.elan.sign_tiers)
            result["signs"] = len(signs)
            result["sign_tier"] = tier
            result["median_sign_duration_s"] = float(np.median([s.duration for s in signs])) if signs else None
            if not signs:
                result["errors"].append("EAF contains zero valid sign segments")
        except Exception as exc:
            result["errors"].append(f"EAF parse error: {exc}")

    text_path = text_cache_directory(cfg) / f"{vid}.npz"
    result["text_cache_path"] = str(text_path)
    if not text_path.is_file():
        result["errors"].append(f"Text feature cache missing: {text_path}")
    else:
        try:
            verify_text_cache(text_path, cfg, vid)
            with np.load(text_path, allow_pickle=False) as cache:
                embeddings = cache["embeddings"]
                result["text_shape"] = list(embeddings.shape)
                if embeddings.ndim != 2 or len(embeddings) != result["subtitle_cues"]:
                    result["errors"].append("Text feature count != VTT cue count")
                if not np.isfinite(embeddings).all():
                    result["errors"].append("Text features contain NaN or Inf")
        except Exception as exc:
            result["errors"].append(f"Text cache error: {exc}")

    return result


def _visual_one_frame(cfg, vid: str, preview_path: Path, device: str) -> dict: