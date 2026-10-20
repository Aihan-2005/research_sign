from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import yaml


@dataclass(frozen=True)
class PathsConfig:
    videos: Path
    poses: Path
    subtitles_raw: Path
    subtitles_gt: Path
    segmentation_gt: Path
    video_ids: Path
    train_video_ids: Path
    val_video_ids: Path
    test_video_ids: Path
    segmentation_output: Path
    alignment_output: Path
    reports: Path
    feature_cache: Path
    text_feature_cache: Path
    checkpoints: Path


@dataclass(frozen=True)
class PoseConfig:
    format: str = "mediapipe"
    num_workers: int = 1
    model_complexity: int = 2
    smooth_landmarks: bool = False
    refine_face_landmarks: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ElanConfig:
    sign_tiers: List[str]
    gt_sign_tiers: List[str]
    aligned_subtitle_tier: str = "SUBTITLE_SHIFTED"


@dataclass(frozen=True)
class SegmentationConfig:
    model: str = "model_E4s-1.pth"
    sign_b_threshold: int = 30
    sign_o_threshold: int = 50
    overwrite: bool = False


@dataclass(frozen=True)
class SegmentationEvalConfig:
    iou_thresholds: List[float]
    boundary_tolerance_ms: List[int]
    tune_b_values: List[int]
    tune_o_values: List[int]
    primary_iou_threshold: float = 0.50


@dataclass(frozen=True)
class ScannerConfig:
    rgb_model: str = "facebook/dinov2-base"
    device: str = "auto"
    amp: bool = True
    rgb_batch_size: int = 32
    samples_per_segment: int = 8
    views: List[str] | None = None
    hand_padding: float = 2.0
    face_padding: float = 1.55
    signer_padding: float = 1.20
    local_fallback_to_signer: bool = True
    pose_include_confidence: bool = True
    pose_shoulder_epsilon: float = 0.02
    cache_dtype: str = "float16"
    overwrite: bool = False

    def resolved_views(self) -> list[str]:
        return self.views or ["signer", "left_hand", "right_hand", "face"]


@dataclass(frozen=True)
class SemanticConfig:
    text_model: str = "BAAI/bge-m3"
    text_backend: str = "bge_m3"  # alternatives: qwen3_embedding
    text_cache_layout: str = "legacy"  # legacy | by_backend
    text_instruction: str = ""  # optional Qwen query instruction
    text_revision: str = "main"  # pin a commit SHA for reproducibility
    text_device: str = "auto"
    text_amp: bool = True
    text_batch_size: int = 16
    text_max_length: int = 256
    text_cache_dtype: str = "float16"
    overwrite_text_cache: bool = False
    embedding_dim: int = 512
    visual_backend: str = "pose_rgb"  # alternatives: pose_only
    search_radius_seconds: float = 6.0
    min_candidate_segments: int = 8
    max_candidate_segments: int = 48
    max_span_segments: int = 16
    max_span_seconds: float = 14.0
    align_temperature: float = 0.07
    temporal_prior_weight: float = 0.35
    temporal_sigma_seconds: float = 3.0
    duration_prior_weight: float = 0.25
    mil_temperature: float = 0.10
    contrastive_temperature: float = 0.07
    supervised_iou_temperature: float = 0.10
    global_top_k: int = 8
    global_decoder: str = "monotonic_dp"
    overlap_penalty: float = 2.0
    gap_change_penalty: float = 0.15


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "auto"
    amp: bool = True
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    contrastive_weight: float = 1.0
    supervised_weight: float = 1.0
    save_name: str = "semantic_aligner.pt"


@dataclass(frozen=True)
class AlignmentConfig:
    duration_penalty_weight: float = 1.0
    gap_penalty_weight: float = 5.0
    max_gap_seconds: float = 10.0
    candidate_window_size: int = 50
    pre_shift_start_seconds: float = 0.0
    pre_shift_end_seconds: float = 0.0
    post_shift_start_seconds: float = 0.0
    post_shift_end_seconds: float = 0.0
    similarity_weight: float = 0.0
    write_updated_eaf: bool = True
    overwrite: bool = False
    # New pluggable, conservative baseline controls (old configs supported).
    backend: str = "sea_temporal"  # alternative: original
    sign_eaf_dir: str | None = None  # optional externally supplied ELAN SIGN tier
    max_boundary_shift_seconds: float = 1.5
    auto_apply_small_shifts: bool = False


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    pose: PoseConfig
    elan: ElanConfig
    segmentation: SegmentationConfig
    segmentation_eval: SegmentationEvalConfig
    scanner: ScannerConfig
    semantic: SemanticConfig
    training: TrainingConfig
    alignment: AlignmentConfig


def _pathify(raw: dict) -> PathsConfig:
    return PathsConfig(**{k: Path(v) for k, v in raw.items()})


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    return AppConfig(
        paths=_pathify(raw["paths"]),
        pose=PoseConfig(**raw.get("pose", {})),
        elan=ElanConfig(**raw["elan"]),
        segmentation=SegmentationConfig(**raw.get("segmentation", {})),
        segmentation_eval=SegmentationEvalConfig(**raw["segmentation_eval"]),
        scanner=ScannerConfig(**raw.get("scanner", {})),
        semantic=SemanticConfig(**raw.get("semantic", {})),
        training=TrainingConfig(**raw.get("training", {})),
        alignment=AlignmentConfig(**raw.get("alignment", {})),
    )


def read_video_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip() and not x.lstrip().startswith("#")]


def model_tag(model: str) -> str:
    stem = Path(model).stem
    return stem[len("model_"):] if stem.startswith("model_") else stem


def segmentation_run_name(model: str, b: int, o: int) -> str:
    return f"{model_tag(model)}_{b}_{o}"


def segmentation_subdir(cfg: AppConfig) -> Path:
    s = cfg.segmentation
    return cfg.paths.segmentation_output / segmentation_run_name(
        s.model, s.sign_b_threshold, s.sign_o_threshold)

