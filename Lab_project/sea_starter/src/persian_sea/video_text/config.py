from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class VideoTextConfig:

    train_ids: str = "data/train_ids.txt"
    val_ids: str = "data/val_ids.txt"
    test_ids: str = "data/test_ids.txt"
    subtitles_dir: str = "data/subtitles_raw"
    approved_subtitles_dir: str = "data/subtitles_approved"
    feature_cache_dir: str = "outputs/features"
    text_cache_dir: str = "outputs/text_features"
    output_dir: str = "outputs/video_text"
    pair_source: str = "raw"
    sea_alignment_dir: str = "outputs/aligned_subtitles"
    semantic_alignment_dir: str = "outputs/aligned_subtitles/semantic"
    max_alignment_shift_seconds: float = 0.0
    visual_backend: str = "pose_rgb"
    embedding_dim: int = 256
    model_dim: int = 384
    temporal_layers: int = 1
    temporal_heads: int = 4
    visual_microbatch: int = 8
    candidate_margin_seconds: float = 4.0
    min_candidate_segments: int = 4
    max_candidate_segments: int = 24
    span_sizes: tuple[int, ...] = (1, 2, 4, 8)
    span_pooling: str = "mean"
    text_backend: str = ""
    text_model: str = ""
    text_instruction: str = ""
    require_text_cache_metadata: bool = False
    mil_temperature: float = 0.10
    contrastive_temperature: float = 0.07
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 8
    max_train_steps: int = 0
    val_max_samples: int = 128
    num_workers: int = 0
    amp: bool = True
    grad_clip_norm: float = 1.0
    seed: int = 42
    checkpoint_name: str = "best_video_text.pt"
    minimum_videos_per_batch: int = 2

    # Positive offset means the sign segment is expected AFTER the subtitle time.
    # Applied only when pair_source='raw'. Corrected/approved alignment sources
    # are assumed to already contain sign-aligned timestamps.
    subtitle_to_sign_offset_seconds: float = 2.0
    subtitle_offsets_by_video: dict[str, float] = field(default_factory=dict)

    def get_subtitle_offset(self, video_id: str) -> float:
        value = self.subtitle_offsets_by_video.get(
            video_id, self.subtitle_to_sign_offset_seconds
        )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite subtitle offset for {video_id}: {value}")
        return value

    def validate(self):
        if self.span_pooling not in {"mean", "attentive"}:
            raise ValueError("span_pooling must be mean or attentive")
        if self.require_text_cache_metadata and (not self.text_backend or not self.text_model):
            raise ValueError("Strict text cache metadata needs text_backend and text_model")
        from persian_sea.alignment_sources import AlignmentSourceSettings
        AlignmentSourceSettings(self.pair_source, self.approved_subtitles_dir,
            self.sea_alignment_dir, self.semantic_alignment_dir,
            self.max_alignment_shift_seconds).validate()
        if not (self.embedding_dim > 0 and self.temporal_layers >= 0 and
                self.temporal_heads > 0 and self.embedding_dim % self.temporal_heads == 0):
            raise ValueError("Bad temporal encoder configuration")
        if not (0 < self.min_candidate_segments <= self.max_candidate_segments):
            raise ValueError("Invalid candidate segment limits")
        if self.candidate_margin_seconds < 0 or any(k <= 0 for k in self.span_sizes):
            raise ValueError("Invalid margin or span_sizes")
        if self.batch_size < self.minimum_videos_per_batch or self.visual_microbatch < 1:
            raise ValueError("Batch must contain at least two distinct videos")
        if not (self.mil_temperature > 0 and self.contrastive_temperature > 0):
            raise ValueError("Temperatures must be positive")
        if self.epochs < 1 or self.val_max_samples < 1:
            raise ValueError("Invalid epochs or validation sample limit")
        if not math.isfinite(float(self.subtitle_to_sign_offset_seconds)):
            raise ValueError("subtitle_to_sign_offset_seconds must be finite")
        if not isinstance(self.subtitle_offsets_by_video, dict):
            raise ValueError("subtitle_offsets_by_video must be a mapping")
        for video_id, offset in self.subtitle_offsets_by_video.items():
            if not isinstance(video_id, str) or not video_id.strip():
                raise ValueError("Per-video offset keys must be nonempty video IDs")
            if not isinstance(offset, (int, float)) or not math.isfinite(float(offset)):
                raise ValueError(f"Invalid per-video subtitle offset: {video_id}={offset!r}")
        return self


def read_config(path: str | Path) -> VideoTextConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("YAML must be a mapping")
    unknown = set(raw) - set(VideoTextConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown video_text config keys: {sorted(unknown)}")
    if "span_sizes" in raw:
        raw["span_sizes"] = tuple(raw["span_sizes"])
    return VideoTextConfig(**raw).validate()
