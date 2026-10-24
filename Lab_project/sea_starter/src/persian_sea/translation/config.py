from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class TranslationConfig:
    visual_config: str = "configs/stage5_mean_bge_pilot.yaml"
    visual_checkpoint: str = "outputs/video_text/stage5_mean_bge_pilot/best_video_text.pt"
    require_pretrained_visual: bool = True
    model_name: str = "google/mt5-small"
    train_ids: str = "data/train_ids.txt"
    val_ids: str = "data/val_ids.txt"
    test_ids: str = "data/test_ids.txt"
    subtitles_dir: str = "data/subtitles_raw"
    visual_cache_dir: str = "outputs/features"
    output_dir: str = "outputs/translation/mt5_first"
    signer_metadata: str = "" 
    strict_signer: bool = False
    max_candidate_segments: int = 48
    min_candidate_segments: int = 4
    cue_margin_seconds: float = 2.0
    max_target_tokens: int = 96
    max_source_segments: int = 32  
    inference_stride_segments: int = 32  
    batch_size: int = 1
    accumulation_steps: int = 4
    epochs: int = 4
    max_train_steps: int = 0  # 0=unlimited
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    freeze_visual: bool = True
    amp: bool = False 
    seed: int = 42
    max_generate_tokens: int = 96

    def validate(self) -> "TranslationConfig":
        if not self.model_name or not self.visual_config:
            raise ValueError("model_name and visual_config are required")
        if not (1 <= self.min_candidate_segments <= self.max_candidate_segments):
            raise ValueError("Bad candidate segment limits")
        if not (0 <= self.cue_margin_seconds <= 120):
            raise ValueError("cue_margin_seconds outside 0..120")
        if not (1 <= self.inference_stride_segments <= self.max_source_segments):
            raise ValueError("Stride must be between 1 and max_source_segments")
        if self.max_source_segments < 1 or self.max_target_tokens < 4:
            raise ValueError("Source/target lengths invalid")
        if self.batch_size < 1 or self.accumulation_steps < 1 or self.epochs < 1:
            raise ValueError("Batch, accumulation and epochs must be positive")
        if self.max_train_steps < 0 or self.learning_rate <= 0:
            raise ValueError("Invalid optimization settings")
        if self.strict_signer and not self.signer_metadata:
            raise ValueError("strict_signer=True requires signer_metadata CSV")
        return self


def read_config(path: str | Path) -> TranslationConfig:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Translation config must be a YAML mapping")
    unknown = set(value) - set(TranslationConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown translation settings: {sorted(unknown)}")
    return TranslationConfig(**value).validate()
