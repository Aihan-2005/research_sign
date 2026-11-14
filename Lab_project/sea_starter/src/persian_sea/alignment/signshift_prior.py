from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math

import numpy as np
import torch


class SignShiftBoundaryRuntime:
    """Load one SignShift checkpoint and produce visual boundary evidence per video.

    The visual cache and EAF SIGN tier must describe the same ordered E4s.
    A strict timestamp check prevents scores being attached to the wrong segment.
    """

    def __init__(self, checkpoint: str | Path, *, time_tolerance_seconds: float = 0.35):
        if not math.isfinite(float(time_tolerance_seconds)) or time_tolerance_seconds < 0:
            raise ValueError("time_tolerance_seconds must be finite and nonnegative")
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SignShift boundary checkpoint not found: {self.checkpoint}")
        self.time_tolerance_seconds = float(time_tolerance_seconds)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from persian_sea.translation.boundary import load_boundary_head
        self.model, self.cfg, self.payload = load_boundary_head(self.checkpoint, self.device)
        if not self.cfg.use_tdm or not self.cfg.use_multistream:
            raise ValueError(
                "Checkpoint is not SignShift TDM+multi-stream. "
                "Train with use_tdm=true and use_multistream=true."
            )

    @torch.inference_mode()
    def predict(self, video_id: str, signs: list) -> dict:
        from persian_sea.translation.boundary import (
            STREAM_NAMES,
            _predict_from_probabilities,
            build_boundary_features_for_inference,
        )

        record = build_boundary_features_for_inference(
            video_id, self.cfg, self.payload, self.device
        )
        cache_starts = np.asarray(record["starts"], dtype=np.float64)
        cache_ends = np.asarray(record["ends"], dtype=np.float64)
        sign_starts = np.asarray([x.start for x in signs], dtype=np.float64)
        sign_ends = np.asarray([x.end for x in signs], dtype=np.float64)

        if len(cache_starts) != len(signs):
            raise ValueError(
                f"{video_id}: visual cache has {len(cache_starts)} segments but "
                f"EAF has {len(signs)} SIGN segments. Rebuild/check the E4s cache."
            )
        if not signs:
            raise ValueError(f"{video_id}: no EAF SIGN segments found")

        deltas = np.maximum(
            np.abs(cache_starts - sign_starts),
            np.abs(cache_ends - sign_ends),
        )
        max_delta = float(deltas.max(initial=0.0))
        if max_delta > self.time_tolerance_seconds:
            bad = int(np.argmax(deltas))
            raise ValueError(
                f"{video_id}: visual-cache/EAF time mismatch {max_delta:.3f}s "
                f"at segment {bad}; tolerance={self.time_tolerance_seconds:.3f}s. "
                "Confirm both were generated from the same E4s segmentation."
            )

        features = torch.as_tensor(record["features"], dtype=torch.float32, device=self.device)
        starts = torch.as_tensor(cache_starts, dtype=torch.float32, device=self.device)
        ends = torch.as_tensor(cache_ends, dtype=torch.float32, device=self.device)
        streams = None
        if self.cfg.use_multistream:
            streams = {
                name: torch.as_tensor(record[name], dtype=torch.float32, device=self.device)
                for name in STREAM_NAMES
            }

        output = self.model(features, starts, ends, streams=streams, return_aux=True)
        probabilities = output["boundary_logits"].sigmoid().float().cpu().numpy()
        windows, count_info = _predict_from_probabilities(
            probabilities, output["count_logits"], record, self.cfg
        )
        return {
            "video_id": video_id,
            "checkpoint": str(self.checkpoint),
            "method": "signshift_tdm_multistream_scp" if self.cfg.use_scp
                      else "signshift_tdm_multistream",
            "num_sign_segments": int(len(signs)),
            "starts": cache_starts,
            "ends": cache_ends,
            "max_cache_eaf_time_delta_seconds": max_delta,
            "start_probs": probabilities[:, 0].astype(np.float64),
            "end_probs": probabilities[:, 1].astype(np.float64),
            "windows": windows,
            "count_prediction": asdict(count_info) if count_info is not None else None,
        }
