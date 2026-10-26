

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import torch

from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.data import VisualCacheLRU, collate, inference_windows, slice_visual
from persian_sea.translation.model import build_model
from persian_sea.translation.train import device_auto, move_tensors


def generate_video(cfg: TranslationConfig, video_id: str, checkpoint: str,
                   output: str | None = None) -> Path:
    path = Path(cfg.visual_cache_dir) / (video_id + ".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Need extracted RGB/Pose visual cache first: {path}")
    d = VisualCacheLRU(1).get(path)
    
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trained_cfg = TranslationConfig(**payload["config"]).validate()
    if trained_cfg != cfg:
        raise ValueError("Inference config differs from saved translation config")
    rgb_dim, views = int(d["rgb"].shape[-1]), int(d["rgb"].shape[-2])
    if list(payload["rgb_dims"]) != [rgb_dim, views]:
        raise ValueError("Incompatible visual dimensions in prediction cache")
    from persian_sea.translation.language import load_tokenizer
    tokenizer_dir = Path(checkpoint).parent / "tokenizer"
    tokenizer = load_tokenizer(cfg, checkpoint_dir=Path(checkpoint).parent)
    model = build_model(replace(cfg, require_pretrained_visual=False), rgb_dim, views)
    model.load_state_dict(payload["model"], strict=True)
    device = device_auto()
    model.to(device).eval()
    predictions = []
    for idx in inference_windows(d, cfg.max_source_segments, cfg.inference_stride_segments):
        item = slice_visual(d, idx)
        batch = collate([item], tokenizer, cfg.max_target_tokens, include_targets=False)
        batch = move_tensors(batch, device)
        ids = model.generate(batch, max_new_tokens=cfg.max_generate_tokens)
        generated = tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
        predictions.append({"start": float(d["starts"][idx[0]]),
                            "end": float(d["ends"][idx[-1]]),
                            "text": generated, "num_sign_segments": len(idx)})
    dest = Path(output) if output else Path(cfg.output_dir) / "predictions" / f"{video_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"video_id": video_id,
                                "source": "visual_cache_only_no_subtitles_no_gloss",
                                "boundary_method": "fixed_segment_windows_not_sentence_detection",
                                "translation_checkpoint": str(checkpoint),
                                "windows": predictions}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return dest
