"""Continuous subtitle-free Persian translation via sentence-boundary detection.

Features come from the cached visual streams only; subtitle text is not required
at inference. Training the boundary model requires approximate subtitle labels.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
import torch

from persian_sea.translation.boundary import (
    extract_segment_features,
    load_boundary_head,
    load_vlp_model,
    predict_windows_with_info,
    sha256_file,
)
from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.data import VisualCacheLRU, collate, slice_visual
from persian_sea.translation.language import load_tokenizer
from persian_sea.translation.model import build_model
from persian_sea.translation.train import device_auto, move_tensors


class ContinuousInferenceEngine:
    def __init__(
        self, cfg: TranslationConfig,
        translation_checkpoint: str | Path,
        boundary_checkpoint: str | Path,
        *,
        vlp_checkpoint: str | Path | None = None,
        device: torch.device | None = None,
    ):
        self.cfg = cfg.validate()
        self.translation_checkpoint = Path(translation_checkpoint)
        self.boundary_checkpoint = Path(boundary_checkpoint)
        if not self.translation_checkpoint.is_file():
            raise FileNotFoundError(
                f"Translation checkpoint not found: {self.translation_checkpoint}"
            )
        if not self.boundary_checkpoint.is_file():
            raise FileNotFoundError(
                f"Boundary checkpoint not found: {self.boundary_checkpoint}"
            )
        self.device = device or device_auto()
        translation_payload = torch.load(
            self.translation_checkpoint, map_location="cpu", weights_only=True
        )
        trained_cfg = TranslationConfig(**translation_payload["config"]).validate()
        if trained_cfg != self.cfg:
            raise ValueError("Translation config differs from saved checkpoint config")

        self.boundary_model, self.boundary_cfg, boundary_payload = load_boundary_head(
            self.boundary_checkpoint, self.device
        )
        chosen_vlp = Path(vlp_checkpoint or boundary_payload["vlp_checkpoint"])
        if not chosen_vlp.is_file():
            raise FileNotFoundError(
                f"VLP checkpoint required by boundary detector not found: {chosen_vlp}. "
                "Pass --vlp-checkpoint if it was moved."
            )
        actual_vlp_hash = sha256_file(chosen_vlp)
        if actual_vlp_hash != boundary_payload["vlp_sha256"]:
            raise ValueError(
                "VLP checkpoint differs from the one used to train the boundary detector"
            )
        translation_vlp = Path(self.cfg.visual_checkpoint)
        if not translation_vlp.is_file():
            raise FileNotFoundError(
                f"Translation config VLP checkpoint not found: {translation_vlp}"
            )
        if sha256_file(translation_vlp) != actual_vlp_hash:
            raise ValueError(
                "Boundary detector and translation checkpoint use different VLP checkpoints. "
                "Retrain the boundary detector using the same visual_checkpoint as translation."
            )
        self.vlp_model, _, _ = load_vlp_model(chosen_vlp, self.device)
        self._cached_vlp_sha256 = actual_vlp_hash

        if (Path(self.cfg.visual_cache_dir).resolve()
                != Path(self.boundary_cfg.visual_cache_dir).resolve()):
            raise ValueError(
                "Translation and boundary configs must use the same visual_cache_dir"
            )
        if (not isinstance(translation_payload.get("rgb_dims"), (list, tuple))
                or len(translation_payload["rgb_dims"]) != 2):
            raise ValueError(
                "Translation checkpoint must contain rgb_dims=[rgb_dim,num_views]"
            )
        rgb_dim, views = [int(x) for x in translation_payload["rgb_dims"]]
        self.tokenizer = load_tokenizer(
            self.cfg, checkpoint_dir=self.translation_checkpoint.parent
        )
        self.translator = build_model(
            replace(self.cfg, require_pretrained_visual=False), rgb_dim, views
        )
        self.translator.load_state_dict(translation_payload["model"], strict=True)
        self.translator.to(self.device).eval()
        self.cache = VisualCacheLRU(1)

    @torch.inference_mode()
    def predict(self, video_id: str) -> dict:
        visual_path = Path(self.boundary_cfg.visual_cache_dir) / f"{video_id}.npz"
        if not visual_path.is_file():
            raise FileNotFoundError(
                f"Visual cache missing for {video_id}: {visual_path}"
            )
        visual = self.cache.get(visual_path)
        feature_record = extract_segment_features(
            video_id, self.vlp_model,
            self.boundary_cfg.visual_cache_dir,
            self.boundary_cfg.feature_dir,
            self._cached_vlp_sha256, self.device,
            self.boundary_cfg.feature_batch_size,
            include_local_views=self.boundary_cfg.use_multistream,
        )
        windows, count_info = predict_windows_with_info(
            self.boundary_model,
            feature_record,
            self.boundary_cfg,
            self.device,
        )
        predictions = []
        for window in windows:
            indices = np.arange(
                window["start_index"], window["end_index"] + 1,
                dtype=np.int64,
            )
            if len(indices) > self.cfg.max_source_segments:
                raise ValueError(
                    f"Predicted sentence has {len(indices)} E4s sign segments, "
                    f"but translation model was configured for max_source_segments="
                    f"{self.cfg.max_source_segments}. Use a boundary config "
                    "max_sentence_segments <= translator max_source_segments, "
                    "or retrain the translator for longer sentences."
                )
            item = slice_visual(visual, indices)
            batch = collate(
                [item], self.tokenizer, self.cfg.max_target_tokens,
                include_targets=False,
                translation_backend=self.cfg.translation_backend,
            )
            batch = move_tensors(batch, self.device)
            ids = self.translator.generate(
                batch, max_new_tokens=self.cfg.max_generate_tokens
            )
            text = self.tokenizer.batch_decode(
                ids, skip_special_tokens=True
            )[0].strip()
            predictions.append({**window, "text": text})
        transcript = "\n".join(row["text"] for row in predictions if row["text"])
        return {
            "video_id": video_id,
            "source": "visual_cache_only_no_subtitles_no_gloss",
            "boundary_method": (
                "signshift_inspired_tdm_scp_count_dp"
                if self.boundary_cfg.use_scp else "weakly_supervised_boundary_detector"
            ),
            "count_prediction": asdict(count_info) if count_info else None,
            "translation_checkpoint": str(self.translation_checkpoint),
            "boundary_checkpoint": str(self.boundary_checkpoint),
            "num_sign_segments": int(len(visual["starts"])),
            "num_predicted_sentences": len(predictions),
            "transcript": transcript,
            "windows": predictions,
            "warning": (
                "Boundary training uses approximate VTT-derived weak labels. "
                "Check segmentation on independently verified videos. "
                "A subtitle cue may not always equal one semantic sentence."
            ),
        }


def write_prediction(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path
