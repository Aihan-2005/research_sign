from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import math
import os
import random
import warnings

import numpy as np
import torch
from torch import nn
import yaml

from persian_sea.io.subtitles import find_subtitle, read_subtitles
from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
from persian_sea.alignment.timing_policy import resolve_offset, shift_cues
from persian_sea.translation.data import VisualCacheLRU, load_ids
from persian_sea.video_text.config import VideoTextConfig
from persian_sea.video_text.model import VideoTextModel


@dataclass(frozen=True)
class BoundaryConfig:
    """Settings for the weakly-supervised boundary detector."""
    train_ids: str = "data/train_ids.txt"
    val_ids: str = "data/val_ids.txt"
    test_ids: str = "data/test_ids.txt"
    subtitles_dir: str = "data/subtitles_raw"
    pair_source: str = "raw"
    approved_subtitles_dir: str = "data/subtitles_approved"
    sea_alignment_dir: str = "outputs/aligned_subtitles"
    semantic_alignment_dir: str = "outputs/aligned_subtitles/semantic"
    max_alignment_shift_seconds: float = 0.0
    require_alignment_review: bool = False
    subtitle_to_sign_offset_seconds: float = 0.0
    subtitle_offsets_by_video: dict[str, float] = field(default_factory=dict)
    visual_cache_dir: str = "outputs/features"
    feature_dir: str = "outputs/boundary_features"
    vlp_checkpoint: str = "outputs/video_text/pretrain_first/best_video_text.pt"
    output_dir: str = "outputs/sentence_boundaries"
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    epochs: int = 20
    patience: int = 5
    threshold: float = 0.35
    min_sentence_segments: int = 2
    max_sentence_segments: int = 48
    feature_batch_size: int = 32
    positive_weight_cap: float = 30.0
    max_train_videos: int = 0
    seed: int = 42

    def validate(self) -> "BoundaryConfig":
        AlignmentSourceSettings(
            mode=self.pair_source,
            approved_dir=self.approved_subtitles_dir,
            sea_dir=self.sea_alignment_dir,
            semantic_dir=self.semantic_alignment_dir,
            max_shift_seconds=self.max_alignment_shift_seconds,
            require_alignment_review=self.require_alignment_review,
        ).validate()
        if not isinstance(self.require_alignment_review, bool):
            raise ValueError("require_alignment_review must be boolean")
        if not math.isfinite(float(self.subtitle_to_sign_offset_seconds)):
            raise ValueError("subtitle_to_sign_offset_seconds must be finite")
        if not isinstance(self.subtitle_offsets_by_video, dict):
            raise ValueError("subtitle_offsets_by_video must be a mapping")
        for video_id, offset in self.subtitle_offsets_by_video.items():
            if not isinstance(video_id, str) or not video_id.strip():
                raise ValueError("Per-video offset keys must be nonempty video IDs")
            if not isinstance(offset, (int, float)) or not math.isfinite(float(offset)):
                raise ValueError(f"Invalid per-video subtitle offset: {video_id}={offset!r}")

        if self.hidden_dim < 8 or self.num_heads < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be >=8 and divisible by num_heads")
        if self.num_layers < 1 or not (0.0 <= self.dropout < 1.0):
            raise ValueError("Invalid transformer layers/dropout")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")
        if not (0.05 <= self.threshold <= 0.95):
            raise ValueError("threshold should be in [0.05, 0.95]")
        if not (1 <= self.min_sentence_segments <= self.max_sentence_segments):
            raise ValueError("Invalid min/max sentence segment limits")
        if self.feature_batch_size < 1 or self.positive_weight_cap < 1:
            raise ValueError("Invalid feature batch size / positive weight cap")
        if self.max_train_videos < 0:
            raise ValueError("max_train_videos cannot be negative")
        return self


def read_boundary_config(path: str | Path) -> BoundaryConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Boundary config must be a YAML mapping")
    unknown = set(raw) - set(BoundaryConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown boundary config keys: {sorted(unknown)}")
    return BoundaryConfig(**raw).validate()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: str | Path, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return payload


def load_vlp_model(checkpoint: str | Path, device: torch.device):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"VLP checkpoint not found: {path}")
    payload = _torch_load(path, "cpu")
    raw_cfg = payload.get("config")
    dims = payload.get("dims")
    if not isinstance(raw_cfg, dict) or not isinstance(dims, (list, tuple)) or len(dims) != 3:
        raise ValueError(
            "VLP checkpoint must contain config and dims=[text_dim,rgb_dim,num_views]"
        )
    raw_cfg = dict(raw_cfg)
    if "span_sizes" in raw_cfg:
        raw_cfg["span_sizes"] = tuple(raw_cfg["span_sizes"])
    cfg = VideoTextConfig(**raw_cfg).validate()
    model = VideoTextModel(
        cfg, text_dim=int(dims[0]), rgb_dim=int(dims[1]), num_views=int(dims[2])
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload, sha256_file(path)


def _valid_feature_cache(
    path: Path, source_path: Path, vlp_hash: str, video_id: str
) -> dict | None:
    if not path.is_file():
        return None
    stat = source_path.stat()
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (
                str(archive["vlp_sha256"].item()) != vlp_hash
                or str(archive["video_id"].item()) != video_id
                or int(archive["source_size"].item()) != stat.st_size
                or int(archive["source_mtime_ns"].item()) != stat.st_mtime_ns
            ):
                return None
            result = {
                key: np.asarray(archive[key])
                for key in ("features", "starts", "ends")
            }
        if (
            result["features"].ndim != 2
            or len(result["features"]) != len(result["starts"])
            or len(result["starts"]) != len(result["ends"])
            or not all(np.isfinite(result[key]).all() for key in result)
            or np.any(result["ends"] <= result["starts"])
        ):
            return None
        return result
    except (OSError, ValueError, KeyError, EOFError):
        return None


def extract_segment_features(
    video_id: str,
    visual_model: VideoTextModel,
    visual_cache_dir: str | Path,
    feature_dir: str | Path,
    vlp_hash: str,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    source = Path(visual_cache_dir) / f"{video_id}.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Visual cache missing for {video_id}: {source}")
    dest_dir = Path(feature_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{video_id}.npz"
    cached = _valid_feature_cache(dest, source, vlp_hash, video_id)
    if cached is not None:
        return cached

    visual = VisualCacheLRU(1).get(source)
    pose = visual["pose"]
    rgb = visual["rgb"]
    features: list[np.ndarray] = []
    segment_encoder = visual_model.video.visual
    segment_encoder.eval()
    with torch.inference_mode():
        for offset in range(0, len(visual["starts"]), batch_size):
            stop = min(len(visual["starts"]), offset + batch_size)
            pose_batch = torch.as_tensor(
                pose[offset:stop], dtype=torch.float32, device=device
            )
            rgb_batch = torch.as_tensor(
                rgb[offset:stop], dtype=torch.float32, device=device
            )
            output = segment_encoder(pose_batch, rgb_batch)
            fused = output.get("fused") if isinstance(output, dict) else None
            if fused is None or fused.ndim != 2:
                raise ValueError(
                    "VLP visual encoder must return a [segments, dim] 'fused' tensor"
                )
            features.append(fused.float().cpu().numpy())

    matrix = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    if len(matrix) != len(visual["starts"]) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid extracted segment features for {video_id}")
    stat = source.stat()
    temp = dest.with_name(dest.name + ".tmp")
    try:
        with temp.open("wb") as file:
            np.savez_compressed(
                file,
                features=matrix,
                starts=visual["starts"].astype(np.float32),
                ends=visual["ends"].astype(np.float32),
                vlp_sha256=np.asarray(vlp_hash),
                source_size=np.asarray(stat.st_size, dtype=np.int64),
                source_mtime_ns=np.asarray(stat.st_mtime_ns, dtype=np.int64),
                video_id=np.asarray(video_id),
            )
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)
    return {
        "features": matrix,
        "starts": visual["starts"],
        "ends": visual["ends"],
    }


def make_boundary_targets(
    starts: np.ndarray, ends: np.ndarray, cues
) -> tuple[np.ndarray, int]:
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if (
        starts.ndim != 1
        or ends.shape != starts.shape
        or len(starts) == 0
        or not np.isfinite(starts).all()
        or not np.isfinite(ends).all()
        or np.any(ends <= starts)
        or np.any(np.diff(starts) < 0)
    ):
        raise ValueError("Invalid segment times for boundary targets")

    centers = (starts + ends) / 2.0
    labels = np.zeros((len(starts), 2), dtype=np.float32)
    valid_cues = 0
    video_start, video_end = float(starts[0]), float(ends[-1])
    for cue in cues:
        if not getattr(cue, "text", "").strip() or cue.end <= cue.start:
            continue
        if cue.end < video_start or cue.start > video_end:
            continue
        cue_start = float(np.clip(cue.start, video_start, video_end))
        cue_end = float(np.clip(cue.end, video_start, video_end))
        if cue_end < cue_start:
            continue
        start_index = int(np.argmin(np.abs(centers - cue_start)))
        end_index = int(np.argmin(np.abs(centers - cue_end)))
        if end_index < start_index:
            continue
        labels[start_index, 0] = 1.0
        labels[end_index, 1] = 1.0
        valid_cues += 1
    if valid_cues == 0 or labels.sum() == 0:
        raise ValueError("No subtitle cues overlap the visual cache; check VTT timebase")
    return labels, valid_cues


def sinusoidal_position_encoding(length: int, dim: int, device, dtype) -> torch.Tensor:
    if length < 1 or dim < 1:
        raise ValueError("Position encoding length and dim must be positive")
    positions = torch.arange(
        length, device=device, dtype=torch.float32
    ).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * scale)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(
            positions * scale[:encoding[:, 1::2].shape[1]]
        )
    return encoding.to(dtype=dtype)


class SentenceBoundaryHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim < 1 or hidden_dim < 8 or hidden_dim % num_heads:
            raise ValueError("Invalid boundary-head dimensions")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, features: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected [N,{self.input_dim}] segment features")
        if starts.shape != (features.shape[0],) or ends.shape != starts.shape:
            raise ValueError("Segment time vectors must have shape [N]")
        if features.shape[0] == 0 or not torch.isfinite(features).all():
            raise ValueError("Empty/nonfinite boundary features")
        duration = (ends - starts).clamp_min(1e-4)
        gap = torch.zeros_like(starts)
        if len(starts) > 1:
            gap[1:] = (starts[1:] - ends[:-1]).clamp_min(0)
        timing = torch.stack(
            [torch.log1p(duration), torch.log1p(gap)], dim=-1
        )
        hidden = self.input_projection(
            torch.cat([features.float(), timing.float()], dim=-1)
        )
        hidden = hidden + sinusoidal_position_encoding(
            len(hidden), self.hidden_dim, hidden.device, hidden.dtype
        )
        hidden = self.norm(
            self.temporal(hidden.unsqueeze(0)).squeeze(0)
        )
        return torch.cat(
            [self.start_head(hidden), self.end_head(hidden)], dim=-1
        )


def threshold_peaks(
    probabilities: np.ndarray, threshold: float, min_separation: int = 2
) -> list[int]:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("Boundary probabilities contain nonfinite values")
    active = np.flatnonzero(scores >= threshold)
    if not len(active):
        return []
    groups: list[list[int]] = [[int(active[0])]]
    for index in active[1:]:
        index = int(index)
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    peaks = [max(group, key=lambda index: scores[index]) for group in groups]
    selected: list[int] = []
    for index in peaks:
        if selected and index - selected[-1] < min_separation:
            if scores[index] > scores[selected[-1]]:
                selected[-1] = index
        else:
            selected.append(index)
    return selected


def pair_boundary_events(
    start_prob: np.ndarray,
    end_prob: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    threshold: float = 0.35,
    min_segments: int = 2,
    max_segments: int = 48,
) -> list[dict]:
    start_prob = np.asarray(start_prob, dtype=np.float64).reshape(-1)
    end_prob = np.asarray(end_prob, dtype=np.float64).reshape(-1)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    count = len(starts)
    if (
        count == 0 or len(ends) != count
        or len(start_prob) != count or len(end_prob) != count
    ):
        raise ValueError("Boundary score/time lengths differ or are empty")
    if not (1 <= min_segments <= max_segments):
        raise ValueError("Invalid sentence length constraints")
    start_events = threshold_peaks(start_prob, threshold)
    end_events = threshold_peaks(end_prob, threshold)

    ending_at: dict[int, list[tuple[int, int, float]]] = {}
    for start_index in start_events:
        for end_index in end_events:
            length = end_index - start_index + 1
            if min_segments <= length <= max_segments:
                score = (
                    0.5 * (start_prob[start_index] + end_prob[end_index])
                    - 0.001 * length
                )
                ending_at.setdefault(end_index, []).append(
                    (start_index, end_index, float(score))
                )

    dp = [0.0] * (count + 1)
    choice: list[tuple[int, int, float] | None] = [None] * (count + 1)
    for prefix in range(1, count + 1):
        dp[prefix] = dp[prefix - 1]
        for candidate in ending_at.get(prefix - 1, []):
            start_index, end_index, score = candidate
            candidate_score = dp[start_index] + score
            if candidate_score > dp[prefix]:
                dp[prefix] = candidate_score
                choice[prefix] = candidate

    selected: list[tuple[int, int, float]] = []
    prefix = count
    while prefix > 0:
        candidate = choice[prefix]
        if candidate is None:
            prefix -= 1
            continue
        start_index, end_index, score = candidate
        if end_index != prefix - 1:
            prefix -= 1
            continue
        selected.append(candidate)
        prefix = start_index

    selected.reverse()
    result: list[dict] = []
    for start_index, end_index, _score in selected:
        result.append(
            {
                "start_index": int(start_index),
                "end_index": int(end_index),
                "start": float(starts[start_index]),
                "end": float(ends[end_index]),
                "start_confidence": float(start_prob[start_index]),
                "end_confidence": float(end_prob[end_index]),
                "num_sign_segments": int(end_index - start_index + 1),
            }
        )
    return result


def _read_optional_ids(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        return []
    return load_ids(path)


def _check_splits(cfg: BoundaryConfig) -> tuple[list[str], list[str]]:
    train_ids = load_ids(cfg.train_ids)
    val_ids = load_ids(cfg.val_ids)
    test_ids = _read_optional_ids(cfg.test_ids)
    sets = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(sets[a] & sets[b])
        if overlap:
            raise ValueError(f"Video IDs overlap between {a} and {b}: {overlap[:8]}")
    return train_ids, val_ids


def _load_record(
    video_id: str,
    cfg: BoundaryConfig,
    model: VideoTextModel,
    vlp_hash: str,
    device: torch.device,
    *,
    extract: bool,
) -> dict:
    if extract:
        data = extract_segment_features(
            video_id, model, cfg.visual_cache_dir, cfg.feature_dir,
            vlp_hash, device, cfg.feature_batch_size,
        )
    else:
        path = Path(cfg.feature_dir) / f"{video_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Boundary feature cache missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["vlp_sha256"].item()) != vlp_hash:
                raise ValueError(
                    f"Feature cache was made with a different VLP checkpoint: {path}"
                )
            data = {
                key: np.asarray(archive[key])
                for key in ("features", "starts", "ends")
            }

    subtitle = find_subtitle(Path(cfg.subtitles_dir), video_id)
    if subtitle is None:
        raise FileNotFoundError(
            f"Raw VTT/SRT missing for boundary labels: {video_id}"
        )

    raw_cues = read_subtitles(subtitle)
    aligned_cues, _ = load_alignment_cues(
        video_id,
        raw_cues,
        AlignmentSourceSettings(
            mode=cfg.pair_source,
            approved_dir=cfg.approved_subtitles_dir,
            sea_dir=cfg.sea_alignment_dir,
            semantic_dir=cfg.semantic_alignment_dir,
            max_shift_seconds=cfg.max_alignment_shift_seconds,
            require_alignment_review=cfg.require_alignment_review,
        ),
    )
    offset = resolve_offset(
        video_id,
        cfg.pair_source,
        cfg.subtitle_to_sign_offset_seconds,
        cfg.subtitle_offsets_by_video,
    )
    target_cues = shift_cues(aligned_cues, offset)
    labels, valid_cues = make_boundary_targets(
        data["starts"], data["ends"], target_cues
    )
    if len(data["features"]) != len(labels):
        raise ValueError(f"Feature/label lengths differ for {video_id}")
    return {
        "video_id": video_id,
        **data,
        "labels": labels,
        "valid_cues": valid_cues,
    }


def _event_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    tolerance_segments: int = 1,
) -> dict[str, float]:
    predicted = threshold_peaks(probabilities, threshold)
    truth = np.flatnonzero(np.asarray(labels) >= 0.5).tolist()
    unmatched = set(truth)
    true_positive = 0
    for prediction in predicted:
        candidates = [
            target for target in unmatched
            if abs(target - prediction) <= tolerance_segments
        ]
        if candidates:
            chosen = min(
                candidates,
                key=lambda target: (abs(target - prediction), target),
            )
            unmatched.remove(chosen)
            true_positive += 1
    false_positive = len(predicted) - true_positive
    false_negative = len(truth) - true_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_events": float(len(predicted)),
        "reference_events": float(len(truth)),
    }


def _evaluate(
    model: SentenceBoundaryHead,
    records: list[dict],
    device: torch.device,
    threshold: float,
) -> dict:
    model.eval()
    starts_stats, ends_stats = [], []
    with torch.inference_mode():
        for row in records:
            features = torch.as_tensor(
                row["features"], dtype=torch.float32, device=device
            )
            starts = torch.as_tensor(
                row["starts"], dtype=torch.float32, device=device
            )
            ends = torch.as_tensor(
                row["ends"], dtype=torch.float32, device=device
            )
            logits = model(features, starts, ends).sigmoid().cpu().numpy()
            starts_stats.append(
                _event_metrics(logits[:, 0], row["labels"][:, 0], threshold)
            )
            ends_stats.append(
                _event_metrics(logits[:, 1], row["labels"][:, 1], threshold)
            )

    def mean(key, rows):
        return float(np.mean([row[key] for row in rows])) if rows else 0.0

    start_f1 = mean("f1", starts_stats)
    end_f1 = mean("f1", ends_stats)
    return {
        "start_precision": mean("precision", starts_stats),
        "start_recall": mean("recall", starts_stats),
        "start_f1": start_f1,
        "end_precision": mean("precision", ends_stats),
        "end_recall": mean("recall", ends_stats),
        "end_f1": end_f1,
        "macro_boundary_f1": (start_f1 + end_f1) / 2.0,
        "videos": len(records),
    }


def train_boundary_detector(cfg: BoundaryConfig) -> Path:
    cfg.validate()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    train_ids, val_ids = _check_splits(cfg)

    if cfg.max_train_videos:
        train_ids = train_ids[:cfg.max_train_videos]
        val_limit = max(1, min(len(val_ids), cfg.max_train_videos // 2))
        val_ids = val_ids[:val_limit]
        warnings.warn(
            f"SMOKE TEST ONLY: using {len(train_ids)} train and "
            f"{len(val_ids)} validation videos"
        )

    if len(train_ids) < 2 or len(val_ids) < 1:
        raise ValueError(
            "Boundary training needs >=2 train videos and >=1 validation video"
        )
    if not Path(cfg.vlp_checkpoint).is_file():
        raise FileNotFoundError(
            f"Train VLP first; checkpoint missing: {cfg.vlp_checkpoint}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        warnings.warn(
            "No CUDA GPU detected. Feature extraction/training may be very slow; "
            "use this only for a smoke test."
        )
    print(
        f"[boundary] loading VLP checkpoint on {device}: {cfg.vlp_checkpoint}",
        flush=True,
    )
    vlp_model, vlp_payload, vlp_hash = load_vlp_model(
        cfg.vlp_checkpoint, device
    )

    all_records: dict[str, dict] = {}
    all_ids = train_ids + val_ids
    for index, video_id in enumerate(all_ids, start=1):
        all_records[video_id] = _load_record(
            video_id, cfg, vlp_model, vlp_hash, device, extract=True
        )
        if index % 10 == 0 or index == len(all_ids):
            print(
                f"[boundary] features/labels {index}/{len(all_ids)} videos",
                flush=True,
            )

    del vlp_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_records = [all_records[video_id] for video_id in train_ids]
    val_records = [all_records[video_id] for video_id in val_ids]
    input_dim = int(train_records[0]["features"].shape[1])
    if any(
        row["features"].shape[1] != input_dim
        for row in all_records.values()
    ):
        raise ValueError("Inconsistent boundary feature dimensions")

    positives = np.sum(
        [row["labels"].sum(axis=0) for row in train_records], axis=0
    )
    total = sum(len(row["labels"]) for row in train_records)
    negative = np.maximum(1.0, total - positives)
    pos_weight = np.minimum(
        negative / np.maximum(positives, 1.0), cfg.positive_weight_cap
    )
    pos_weight_t = torch.as_tensor(
        pos_weight, dtype=torch.float32, device=device
    )
    model = SentenceBoundaryHead(
        input_dim, cfg.hidden_dim, cfg.num_layers, cfg.num_heads, cfg.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_sentence_boundary.pt"
    latest_path = output_dir / "last_sentence_boundary.pt"
    best_score, stale = -1.0, 0
    history = output_dir / "history.jsonl"
    if history.exists():
        history.unlink()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        rng = np.random.default_rng(cfg.seed + epoch)
        order = rng.permutation(len(train_records))
        losses = []
        for item in order:
            row = train_records[int(item)]
            features = torch.as_tensor(
                row["features"], dtype=torch.float32, device=device
            )
            starts = torch.as_tensor(
                row["starts"], dtype=torch.float32, device=device
            )
            ends = torch.as_tensor(
                row["ends"], dtype=torch.float32, device=device
            )
            labels = torch.as_tensor(
                row["labels"], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, starts, ends)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite boundary loss on {row['video_id']}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = _evaluate(model, val_records, device, cfg.threshold)
        history_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
        }
        with history.open("a", encoding="utf-8") as file:
            file.write(json.dumps(history_row, ensure_ascii=False) + "\n")
        print(
            "[boundary] " + json.dumps(history_row, ensure_ascii=False),
            flush=True,
        )

        payload = {
            "schema_version": 1,
            "model": model.state_dict(),
            "config": asdict(cfg),
            "input_dim": input_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "dropout": cfg.dropout,
            "threshold": cfg.threshold,
            "vlp_checkpoint": str(Path(cfg.vlp_checkpoint).resolve()),
            "vlp_sha256": vlp_hash,
            "vlp_config": vlp_payload["config"],
            "vlp_dims": [int(value) for value in vlp_payload["dims"]],
            "epoch": epoch,
            "validation": metrics,
            "label_source": "subtitle_timestamps_nearest_segment_center",
            "pair_source": cfg.pair_source,
            "subtitle_to_sign_offset_seconds": float(
                cfg.subtitle_to_sign_offset_seconds
            ),
            "subtitle_offsets_by_video": dict(cfg.subtitle_offsets_by_video),
        }

        temp = latest_path.with_name(latest_path.name + ".tmp")
        try:
            torch.save(payload, temp)
            os.replace(temp, latest_path)
        finally:
            temp.unlink(missing_ok=True)

        score = metrics["macro_boundary_f1"]
        if score > best_score:
            best_score, stale = score, 0
            temp = best_path.with_name(best_path.name + ".tmp")
            try:
                torch.save(payload, temp)
                os.replace(temp, best_path)
            finally:
                temp.unlink(missing_ok=True)
        else:
            stale += 1
            if stale >= cfg.patience:
                print(
                    f"[boundary] early stopping after {epoch} epochs",
                    flush=True,
                )
                break

    if not best_path.is_file():
        raise RuntimeError("Boundary training did not save a best checkpoint")
    return best_path


def load_boundary_head(checkpoint: str | Path, device: torch.device):
    payload = _torch_load(checkpoint, "cpu")
    required = {"model", "config", "input_dim", "vlp_checkpoint", "vlp_sha256"}
    if not required.issubset(payload):
        raise ValueError(
            f"Boundary checkpoint missing keys: {sorted(required - set(payload))}"
        )
    cfg = BoundaryConfig(**payload["config"]).validate()
    model = SentenceBoundaryHead(
        int(payload["input_dim"]),
        cfg.hidden_dim,
        cfg.num_layers,
        cfg.num_heads,
        cfg.dropout,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, cfg, payload


def predict_windows(
    model: SentenceBoundaryHead,
    record: dict,
    cfg: BoundaryConfig,
    device: torch.device,
) -> list[dict]:
    with torch.inference_mode():
        features = torch.as_tensor(
            record["features"], dtype=torch.float32, device=device
        )
        starts = torch.as_tensor(
            record["starts"], dtype=torch.float32, device=device
        )
        ends = torch.as_tensor(
            record["ends"], dtype=torch.float32, device=device
        )
        probabilities = model(features, starts, ends).sigmoid().cpu().numpy()

    return pair_boundary_events(
        probabilities[:, 0],
        probabilities[:, 1],
        record["starts"],
        record["ends"],
        threshold=cfg.threshold,
        min_segments=cfg.min_sentence_segments,
        max_segments=cfg.max_sentence_segments,
    )
