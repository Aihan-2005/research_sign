
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from persian_sea.video_text.config import read_config as read_video_config
from persian_sea.video_text.model import VideoTextModel
from persian_sea.translation.config import TranslationConfig


class VisualLanguageAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(input_dim),
                                     nn.Linear(input_dim, hidden_dim),
                                     nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, hidden_dim))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or not mask.any(dim=1).all():
            raise ValueError("Expected [B,M,D] tokens and nonempty [B,M] mask")
        return self.network(tokens) * mask[..., None].to(tokens.dtype)


class PersianTranslator(nn.Module):
    def __init__(self, visual_tower: nn.Module, visual_dim: int, seq2seq: nn.Module,
                 freeze_visual: bool = True):
        super().__init__()
        self.visual = visual_tower
        self.freeze_visual = bool(freeze_visual)
        if self.freeze_visual:
            for param in self.visual.parameters():
                param.requires_grad_(False)
            self.visual.eval()
        self.decoder = seq2seq
        lm_dim = getattr(seq2seq.config, "d_model", None)
        if not isinstance(lm_dim, int) or lm_dim <= 0:
            raise ValueError("Seq2seq model requires a positive config.d_model")
        self.adapter = VisualLanguageAdapter(visual_dim, lm_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual:
            self.visual.eval()
        return self

    def visual_memory(self, batch: dict):
        context = torch.no_grad() if self.freeze_visual else nullcontext()
        with context:
            video_tokens = self.visual(batch["pose"], batch["rgb"],
                                       batch["starts"], batch["ends"], batch["mask"])
        memory = self.adapter(video_tokens, batch["mask"])
        return memory

    def _encoder_outputs(self, memory: torch.Tensor):
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=memory)

    def forward(self, batch: dict):
        if "labels" not in batch:
            raise ValueError("Training requires target text labels; inference does not")
        memory = self.visual_memory(batch)
        return self.decoder(encoder_outputs=self._encoder_outputs(memory),
                            attention_mask=batch["mask"].long(), labels=batch["labels"])

    @torch.no_grad()
    def generate(self, batch: dict, max_new_tokens: int = 96):
        memory = self.visual_memory(batch)
        return self.decoder.generate(encoder_outputs=self._encoder_outputs(memory),
                                     attention_mask=batch["mask"].long(),
                                     max_new_tokens=max_new_tokens)


def load_visual_tower(cfg: TranslationConfig, rgb_dim: int, views: int):
    """Load the exact pretraining checkpoint architecture; fail on mismatch."""
    vt_cfg = read_video_config(cfg.visual_config)
    if cfg.require_pretrained_visual:
        path = Path(cfg.visual_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing VLP checkpoint: {path}. Train persian-sea-train-video-text first."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        stored = ckpt.get("model_cfg")
        if not isinstance(stored, dict):
            raise ValueError("Checkpoint has no model_cfg provenance")
        from persian_sea.video_text.train import architecture_config, checkpoint_architecture
        if checkpoint_architecture(stored) != architecture_config(vt_cfg):
            raise ValueError("VLP checkpoint architecture differs from visual_config")
        dims = ckpt.get("dims")
        if not isinstance(dims, (list, tuple)) or len(dims) != 3:
            raise ValueError("VLP checkpoint needs [text_dim, rgb_dim, num_views]")
        text_dim, trained_rgb_dim, trained_views = [int(n) for n in dims]
        if (trained_rgb_dim, trained_views) != (rgb_dim, views):
            raise ValueError(f"Visual cache dims {(rgb_dim, views)} differ from checkpoint "
                             f"{(trained_rgb_dim, trained_views)}")
    else:
        # Explicit experimental escape hatch for synthetic smoke tests only.
        text_dim = 1024
        ckpt = None
    model = VideoTextModel(vt_cfg, text_dim=text_dim, rgb_dim=rgb_dim, num_views=views)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=True)
    return model.video, vt_cfg.embedding_dim


def build_model(cfg: TranslationConfig, rgb_dim: int, views: int):
    from transformers import AutoModelForSeq2SeqLM
    visual, dim = load_visual_tower(cfg, rgb_dim, views)
    seq2seq = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    return PersianTranslator(visual, dim, seq2seq, cfg.freeze_visual)
