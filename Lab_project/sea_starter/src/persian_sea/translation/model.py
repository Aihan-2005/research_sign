from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from persian_sea.video_text.config import read_config as read_video_config
from persian_sea.video_text.model import VideoTextModel
from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.ot_alignment import PartialOTAlignmentLoss


class VisualLanguageAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if (
            tokens.ndim != 3
            or mask.shape != tokens.shape[:2]
            or not mask.any(dim=1).all()
        ):
            raise ValueError("Expected [B,M,D] tokens and nonempty [B,M] mask")
        return self.network(tokens) * mask[..., None].to(tokens.dtype)


class PersianTranslator(nn.Module):
    def __init__(
        self,
        visual_tower: nn.Module,
        visual_dim: int,
        seq2seq: nn.Module,
        freeze_visual: bool = True,
        *,
        ota_enabled: bool = False,
        ota_weight: float = 0.10,
        ota_epsilon: float = 0.07,
        ota_iterations: int = 30,
        ota_dustbin_cost: float = 0.35,
        ota_temporal_weight: float = 0.01,
        ota_detach_text: bool = False,
        prior_teacher: nn.Module | None = None,
        tpa_weight: float = 0.05,
        tpa_temperature: float = 1.0,
        tpa_confidence_power: float = 1.0,
    ):
        super().__init__()
        self.visual = visual_tower
        self.freeze_visual = bool(freeze_visual)
        if self.freeze_visual:
            for parameter in self.visual.parameters():
                parameter.requires_grad_(False)
            self.visual.eval()

        self.decoder = seq2seq
        lm_dim = getattr(seq2seq.config, "d_model", None)
        if not isinstance(lm_dim, int) or lm_dim <= 0:
            raise ValueError("Seq2seq model requires a positive config.d_model")
        self.adapter = VisualLanguageAdapter(visual_dim, lm_dim)

        self.ota_enabled = bool(ota_enabled)
        self.ota_weight = float(ota_weight)
        self.ota_temporal_weight = float(ota_temporal_weight)
        self.ota_detach_text = bool(ota_detach_text)
        self.ota_criterion = (
            PartialOTAlignmentLoss(
                epsilon=ota_epsilon,
                iterations=ota_iterations,
                dustbin_cost=ota_dustbin_cost,
            )
            if self.ota_enabled
            else None
        )

        self.prior_teacher = prior_teacher
        self.tpa_weight = float(tpa_weight)
        self.tpa_temperature = float(tpa_temperature)
        self.tpa_confidence_power = float(tpa_confidence_power)
        if self.prior_teacher is not None:
            for parameter in self.prior_teacher.parameters():
                parameter.requires_grad_(False)
            self.prior_teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual:
            self.visual.eval()
        if self.prior_teacher is not None:
            self.prior_teacher.eval()
        return self

    def visual_memory(self, batch: dict):
        context = torch.no_grad() if self.freeze_visual else nullcontext()
        with context:
            video_tokens = self.visual(
                batch["pose"],
                batch["rgb"],
                batch["starts"],
                batch["ends"],
                batch["mask"],
            )
        return self.adapter(video_tokens, batch["mask"])

    def _encoder_outputs(self, memory: torch.Tensor):
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=memory)

    def _compute_tpa_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.prior_teacher is None:
            return student_logits.float().sum() * 0.0

        pad_token_id = getattr(self.prior_teacher.config, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("TPA teacher requires config.pad_token_id")

        batch_size = labels.shape[0]
        neutral_source = torch.full(
            (batch_size, 1),
            int(pad_token_id),
            dtype=torch.long,
            device=labels.device,
        )
        neutral_attention = torch.ones_like(neutral_source)


        with torch.no_grad():
            teacher_output = self.prior_teacher(
                input_ids=neutral_source,
                attention_mask=neutral_attention,
                labels=labels,
                use_cache=False,
            )
            teacher_logits = teacher_output.logits.float()

        temperature = self.tpa_temperature
        student_log_probs = F.log_softmax(
            student_logits.float() / temperature, dim=-1
        )
        student_probs = student_log_probs.exp()
        teacher_log_probs = F.log_softmax(
            teacher_logits / temperature, dim=-1
        )
        teacher_probs = teacher_log_probs.exp()


        per_token_kl = (
            student_probs * (student_log_probs - teacher_log_probs)
        ).sum(dim=-1)
        confidence = teacher_probs.max(dim=-1).values.pow(
            self.tpa_confidence_power
        )
        valid = labels.ne(-100).to(confidence.dtype)
        weights = confidence * valid
        denominator = weights.sum().clamp_min(1e-8)
        return (
            (per_token_kl * weights).sum()
            / denominator
            * (temperature ** 2)
        )

    def forward(self, batch: dict):
        if "labels" not in batch:
            raise ValueError("Training requires target text labels; inference does not")

        memory = self.visual_memory(batch)
        result = self.decoder(
            encoder_outputs=self._encoder_outputs(memory),
            attention_mask=batch["mask"].long(),
            labels=batch["labels"],
        )
        ce_loss = result.loss.float()
        total_loss = ce_loss
        zero = ce_loss.new_zeros(())

        ota_loss = zero
        temporal_loss = zero
        null_ratio = zero
        matched_ratio = zero
        if self.ota_enabled:
            required = (
                "target_input_ids",
                "target_attention_mask",
                "target_content_mask",
            )
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(
                    "OTA training batch is missing token-level target fields: "
                    + ", ".join(missing)
                )

            text_encoder = self.decoder.get_encoder()
            text_output = text_encoder(
                input_ids=batch["target_input_ids"],
                attention_mask=batch["target_attention_mask"],
                return_dict=True,
            )
            text_tokens = text_output.last_hidden_state
            if self.ota_detach_text:
                text_tokens = text_tokens.detach()

            alignment = self.ota_criterion(
                memory,
                text_tokens,
                batch["mask"],
                batch["target_content_mask"],
            )
            ota_loss = alignment["transport_cost"]
            temporal_loss = alignment["temporal_variation"]
            null_ratio = alignment["null_ratio"]
            matched_ratio = alignment["matched_ratio"]
            total_loss = (
                total_loss
                + self.ota_weight * ota_loss
                + self.ota_temporal_weight * temporal_loss
            )

        tpa_loss = zero
        if self.prior_teacher is not None:
            tpa_loss = self._compute_tpa_loss(result.logits, batch["labels"])
            total_loss = total_loss + self.tpa_weight * tpa_loss


        result.loss = total_loss
        result.ce_loss = ce_loss
        result.ota_loss = ota_loss
        result.temporal_variation_loss = temporal_loss
        result.tpa_loss = tpa_loss
        result.null_ratio = null_ratio
        result.matched_ratio = matched_ratio
        return result

    @torch.no_grad()
    def generate(self, batch: dict, max_new_tokens: int = 96):
        memory = self.visual_memory(batch)
        return self.decoder.generate(
            encoder_outputs=self._encoder_outputs(memory),
            attention_mask=batch["mask"].long(),
            max_new_tokens=max_new_tokens,
        )


def load_visual_tower(cfg: TranslationConfig, rgb_dim: int, views: int):
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
        text_dim, trained_rgb_dim, trained_views = [int(value) for value in dims]
        if (trained_rgb_dim, trained_views) != (rgb_dim, views):
            raise ValueError(
                f"Visual cache dims {(rgb_dim, views)} differ from checkpoint "
                f"{(trained_rgb_dim, trained_views)}"
            )
    else:
        text_dim = 1024
        ckpt = None

    model = VideoTextModel(
        vt_cfg,
        text_dim=text_dim,
        rgb_dim=rgb_dim,
        num_views=views,
    )
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=True)
    return model.video, vt_cfg.embedding_dim


def build_model(cfg: TranslationConfig, rgb_dim: int, views: int):
    from transformers import AutoModelForSeq2SeqLM

    visual, dim = load_visual_tower(cfg, rgb_dim=rgb_dim, views=views)
    seq2seq = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    prior_teacher = None
    if cfg.tpa_enabled:
        prior_teacher = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    if cfg.translation_backend == "mbart50":
        from persian_sea.translation.language import (
            load_tokenizer,
            configure_generation,
        )
        tokenizer = load_tokenizer(cfg)
        seq2seq = configure_generation(seq2seq, tokenizer, cfg)
        if prior_teacher is not None:
            prior_teacher = configure_generation(prior_teacher, tokenizer, cfg)

    return PersianTranslator(
        visual,
        dim,
        seq2seq,
        freeze_visual=cfg.freeze_visual,
        ota_enabled=cfg.ota_enabled,
        ota_weight=cfg.ota_weight,
        ota_epsilon=cfg.ota_epsilon,
        ota_iterations=cfg.ota_iterations,
        ota_dustbin_cost=cfg.ota_dustbin_cost,
        ota_temporal_weight=cfg.ota_temporal_weight,
        ota_detach_text=cfg.ota_detach_text,
        prior_teacher=prior_teacher,
        tpa_weight=cfg.tpa_weight,
        tpa_temperature=cfg.tpa_temperature,
        tpa_confidence_power=cfg.tpa_confidence_power,
    )
