from __future__ import annotations

from pathlib import Path
import json
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from persian_sea.config import AppConfig
from persian_sea.semantic.dataset import SubtitleAlignmentDataset, collate_alignment_batch
from persian_sea.semantic.losses import contrastive_bag_loss, supervised_span_loss
from persian_sea.semantic.model import SemanticAligner
from persian_sea.text.cache import TextFeatureCache, text_cache_directory, verify_text_cache
from persian_sea.features.cache import FeatureCache


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def infer_dimensions(cfg: AppConfig, video_ids_path: Path) -> tuple[int, int, int]:
    ids = [x.strip() for x in video_ids_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for video_id in ids:
        visual_path = cfg.paths.feature_cache / f"{video_id}.npz"
        text_path = text_cache_directory(cfg) / f"{video_id}.npz"
        if visual_path.exists() and text_path.exists():
            visual = FeatureCache(visual_path)
            verify_text_cache(text_path, cfg, video_id)
            text = TextFeatureCache(text_path)
            return (
                int(text.embeddings.shape[-1]),
                int(visual.rgb.shape[-1]),
                int(visual.rgb.shape[2]),
            )
    raise RuntimeError("Cannot infer feature dimensions: feature caches not found")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_loss(model: SemanticAligner, batch: dict,
                 cfg: AppConfig) -> tuple[torch.Tensor, dict]:
    output = model(
        batch,
        max_span_segments=cfg.semantic.max_span_segments,
        max_span_seconds=cfg.semantic.max_span_seconds,
        align_temperature=cfg.semantic.align_temperature,
        temporal_prior_weight=cfg.semantic.temporal_prior_weight,
        temporal_sigma_seconds=cfg.semantic.temporal_sigma_seconds,
        duration_prior_weight=cfg.semantic.duration_prior_weight,
    )
    contrastive, _ = contrastive_bag_loss(
        output["text_vectors"], output["samples"], batch["video_id"],
        mil_temperature=cfg.semantic.mil_temperature,
        contrastive_temperature=cfg.semantic.contrastive_temperature,
        rough_start=batch["rough_start"], rough_end=batch["rough_end"],
    )
    supervised, n_sup = supervised_span_loss(
        output["samples"], batch["gt_start"], batch["gt_end"],
        iou_temperature=cfg.semantic.supervised_iou_temperature,
    )
    total = (
        cfg.training.contrastive_weight * contrastive
        + cfg.training.supervised_weight * supervised
    )
    return total, {
        "loss": float(total.detach().item()),
        "contrastive": float(contrastive.detach().item()),
        "supervised": float(supervised.detach().item()),
        "supervised_count": n_sup,
    }


@torch.no_grad()
def evaluate_loss(model, loader, cfg, device) -> dict:
    model.eval()
    sums = {"loss": 0.0, "contrastive": 0.0, "supervised": 0.0}
    batches, supervised_count = 0, 0
    for batch in loader:
        loss, metrics = forward_loss(model, move_batch(batch, device), cfg)
        for key in sums:
            sums[key] += metrics[key]
        supervised_count += metrics["supervised_count"]
        batches += 1
    if not batches:
        return {**{k: float("inf") for k in sums}, "supervised_count": 0}
    return {
        **{k: v / batches for k, v in sums.items()},
        "supervised_count": supervised_count,
    }


def train_semantic(cfg: AppConfig) -> Path:
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    train_set = SubtitleAlignmentDataset(cfg, cfg.paths.train_video_ids)
    val_set = SubtitleAlignmentDataset(cfg, cfg.paths.val_video_ids)

    if len({r.video_id for r in train_set.refs}) < 2:
        warnings.warn(
            "Training uses fewer than two videos. Most same-video negatives "
            "will be masked, so contrastive training may be weak.",
            stacklevel=2,
        )

    train_loader = DataLoader(
        train_set, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, collate_fn=collate_alignment_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, collate_fn=collate_alignment_batch,
        pin_memory=device.type == "cuda",
    )

    text_dim, rgb_dim, num_views = infer_dimensions(cfg, cfg.paths.train_video_ids)
    model = SemanticAligner(
        text_input_dim=text_dim, rgb_input_dim=rgb_dim,
        num_views=num_views, embedding_dim=cfg.semantic.embedding_dim,
        visual_backend=cfg.semantic.visual_backend,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    use_amp = cfg.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint = cfg.paths.checkpoints / cfg.training.save_name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        running = []
        progress = tqdm(train_loader, desc=f"train {epoch}/{cfg.training.epochs}")
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp,
            ):
                loss, metrics = forward_loss(model, batch, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss detected")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running.append(metrics["loss"])
            progress.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                ctr=f"{metrics['contrastive']:.3f}",
                sup=f"{metrics['supervised']:.3f}",
            )
        val = evaluate_loss(model, val_loader, cfg, device)
        mean_train = float(np.mean(running)) if running else float("inf")
        history.append({"epoch": epoch, "train_loss": mean_train, "val": val})
        print(f"[epoch {epoch}] train={mean_train:.4f} val={val['loss']:.4f}")
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "text_dim": text_dim, "rgb_dim": rgb_dim,
                "num_views": num_views, "embedding_dim": cfg.semantic.embedding_dim,
                    "visual_backend": cfg.semantic.visual_backend,
                "val": val,
            }, checkpoint)
            print(f"[checkpoint] saved {checkpoint}")

    history_path = cfg.paths.reports / "semantic_training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return checkpoint

