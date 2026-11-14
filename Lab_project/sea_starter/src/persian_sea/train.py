from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import random
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import VideoTextConfig
from .dataset import (
    VideoTextDataset, VideoDiverseBatchSampler, collate_video_text,
    check_disjoint_splits,
)
from .model import VideoTextModel, mil_similarity_matrix
from .losses import (
    contrastive_loss, partial_ot_similarity_matrix, retrieval_diagnostic,
)


MODEL_KEYS = (
    "visual_backend", "embedding_dim", "model_dim", "temporal_layers",
    "temporal_heads", "span_sizes", "span_pooling", "text_representation",
    "sentence_loss_weight", "word_loss_weight", "word_ot_epsilon",
    "word_ot_iterations", "word_ot_dustbin_score",
)

LEGACY_MODEL_DEFAULTS = {
    "span_pooling": "mean",
    "text_representation": "sentence",
    "sentence_loss_weight": 1.0,
    "word_loss_weight": 1.0,
    "word_ot_epsilon": 0.05,
    "word_ot_iterations": 30,
    "word_ot_dustbin_score": 0.0,
}


def architecture_config(cfg: VideoTextConfig) -> dict:
    raw = asdict(cfg)
    return {key: raw[key] for key in MODEL_KEYS}


def checkpoint_architecture(raw: dict) -> dict:
    result = dict(raw)
    for key, value in LEGACY_MODEL_DEFAULTS.items():
        result.setdefault(key, value)
    return {key: result.get(key) for key in MODEL_KEYS}


def verify_checkpoint_text_source(
    stored_cfg: dict, cfg: VideoTextConfig
) -> None:
    keys = (
        "text_backend", "text_model", "text_instruction",
        "text_revision", "text_max_length", "require_text_cache_metadata",
        "pair_source", "approved_subtitles_dir", "sea_alignment_dir",
        "semantic_alignment_dir", "max_alignment_shift_seconds",
        "subtitle_to_sign_offset_seconds", "subtitle_offsets_by_video",
    )
    defaults = {
        "text_backend": "",
        "text_model": "",
        "text_instruction": "",
        "text_revision": "main",
        "text_max_length": 256,
        "require_text_cache_metadata": False,
        "pair_source": "raw",
        "approved_subtitles_dir": "data/subtitles_approved",
        "sea_alignment_dir": "outputs/aligned_subtitles",
        "semantic_alignment_dir": "outputs/aligned_subtitles/semantic",
        "max_alignment_shift_seconds": 0.0,
        "subtitle_to_sign_offset_seconds": 0.0,
        "subtitle_offsets_by_video": {},
    }
    for key in keys:
        if stored_cfg.get(key, defaults[key]) != getattr(cfg, key):
            raise ValueError(f"Checkpoint text/source provenance differs: {key}")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def dims(dataset: VideoTextDataset):
    item = dataset[0]
    return (
        item["text_embedding"].numel(),
        item["rgb"].shape[-1],
        item["rgb"].shape[2],
    )


def _selection(dataset: VideoTextDataset, limit: int, seed: int):
    rng = np.random.default_rng(seed)
    ids = {}
    for i, ref in enumerate(dataset.refs):
        ids.setdefault(ref.video_id, []).append(i)
    pools = {key: list(rng.permutation(values)) for key, values in ids.items()}
    keys = sorted(pools)
    result = []
    while len(result) < limit:
        any_added = False
        for key in keys:
            if pools[key] and len(result) < limit:
                result.append(pools[key].pop())
                any_added = True
        if not any_added:
            break
    return result


def _pad_ragged(vectors: list[torch.Tensor], device):
    if not vectors:
        raise ValueError("Cannot pad an empty vector collection")
    dim = vectors[0].shape[-1]
    if any(item.ndim != 2 or item.shape[-1] != dim or item.shape[0] < 1
           for item in vectors):
        raise ValueError("Ragged sequences must be nonempty [length, dim] tensors")
    max_length = max(item.shape[0] for item in vectors)
    padded = torch.zeros(len(vectors), max_length, dim, dtype=torch.float32, device=device)
    mask = torch.zeros(len(vectors), max_length, dtype=torch.bool, device=device)
    for i, item in enumerate(vectors):
        count = item.shape[0]
        padded[i, :count] = item.to(device=device, dtype=torch.float32)
        mask[i, :count] = True
    return padded, mask


def _word_score_matrix(
    word_vectors, word_mask, span_vectors, span_mask, cfg: VideoTextConfig
):
    blocks = []
    chunk = cfg.word_eval_query_chunk
    for start in range(0, word_vectors.shape[0], chunk):
        blocks.append(
            partial_ot_similarity_matrix(
                word_vectors[start:start + chunk],
                word_mask[start:start + chunk],
                span_vectors,
                span_mask,
                epsilon=cfg.word_ot_epsilon,
                iterations=cfg.word_ot_iterations,
                dustbin_score=cfg.word_ot_dustbin_score,
            )
        )
    return torch.cat(blocks, dim=0)


@torch.no_grad()
def evaluate(
    model: VideoTextModel,
    dataset: VideoTextDataset,
    cfg: VideoTextConfig,
    device,
):
    model.eval()
    ids = _selection(dataset, cfg.val_max_samples, cfg.seed)
    loader = DataLoader(
        Subset(dataset, ids),
        batch_size=max(1, min(cfg.batch_size, 4)),
        shuffle=False,
        collate_fn=collate_video_text,
        num_workers=0,
    )

    sentence_vectors = []
    word_sequences = []
    span_sequences = []
    videos, text_keys = [], []

    for batch in loader:
        data = move(batch, device)
        output = model(data)
        if "text_vectors" in output:
            sentence_vectors.append(output["text_vectors"].detach().cpu())
        if "word_vectors" in output:
            for i in range(output["word_vectors"].shape[0]):
                valid = output["word_mask"][i]
                word_sequences.append(output["word_vectors"][i, valid].detach().cpu())
        for i in range(output["span_vectors"].shape[0]):
            valid = output["span_mask"][i]
            span_sequences.append(output["span_vectors"][i, valid].detach().cpu())
        videos.extend(batch["video_id"])
        text_keys.extend(batch["text_key"])

    if not span_sequences:
        raise ValueError("Empty validation set")
    span_vectors, span_mask = _pad_ragged(span_sequences, device)
    score_matrices = {}

    if cfg.text_representation in {"sentence", "both"}:
        vectors = torch.cat(sentence_vectors, dim=0).to(device)
        score_matrices["sentence"] = mil_similarity_matrix(
            vectors, span_vectors, span_mask, cfg.mil_temperature
        )

    if cfg.text_representation in {"word", "both"}:
        words, word_mask = _pad_ragged(word_sequences, device)
        score_matrices["word"] = _word_score_matrix(
            words, word_mask, span_vectors, span_mask, cfg
        )

    if cfg.text_representation == "sentence":
        full = score_matrices["sentence"]
    elif cfg.text_representation == "word":
        full = score_matrices["word"]
    else:
        total_weight = cfg.sentence_loss_weight + cfg.word_loss_weight
        full = (
            cfg.sentence_loss_weight * score_matrices["sentence"]
            + cfg.word_loss_weight * score_matrices["word"]
        ) / total_weight

    component_validation_losses = {}
    if cfg.text_representation == "both":
        sentence_loss, extra = contrastive_loss(
            score_matrices["sentence"], videos, text_keys,
            cfg.contrastive_temperature,
        )
        word_loss, _ = contrastive_loss(
            score_matrices["word"], videos, text_keys,
            cfg.contrastive_temperature,
        )
        total_weight = cfg.sentence_loss_weight + cfg.word_loss_weight
        loss = (
            cfg.sentence_loss_weight * sentence_loss
            + cfg.word_loss_weight * word_loss
        ) / total_weight
        component_validation_losses = {
            "sentence_retrieval_loss": float(sentence_loss.detach()),
            "word_retrieval_loss": float(word_loss.detach()),
        }
    else:
        loss, extra = contrastive_loss(
            full, videos, text_keys, cfg.contrastive_temperature
        )
    diagnostic = retrieval_diagnostic(full, videos, text_keys)
    result = {
        "retrieval_loss": float(loss.detach()),
        "r1": diagnostic["r1"],
        "r5": diagnostic["r5"],
        "queries": diagnostic["queries"],
        "queries_at5": diagnostic["queries_at5"],
        "num_pairs": len(videos),
        "negatives_masked": extra["masked_pairs"],
        "text_representation": cfg.text_representation,
        **component_validation_losses,
    }
    for name, matrix in score_matrices.items():
        result[f"{name}_mean_score"] = float(matrix.mean().detach())
    return result


def _training_objective(output, batch, cfg: VideoTextConfig):
    losses = {}
    infos = []

    if cfg.text_representation in {"sentence", "both"}:
        sentence_matrix = mil_similarity_matrix(
            output["text_vectors"], output["span_vectors"],
            output["span_mask"], cfg.mil_temperature,
        )
        sentence_loss, info = contrastive_loss(
            sentence_matrix, batch["video_id"], batch["text_key"],
            cfg.contrastive_temperature,
        )
        losses["sentence"] = sentence_loss
        infos.append(info)

    if cfg.text_representation in {"word", "both"}:
        word_matrix = partial_ot_similarity_matrix(
            output["word_vectors"], output["word_mask"],
            output["span_vectors"], output["span_mask"],
            epsilon=cfg.word_ot_epsilon,
            iterations=cfg.word_ot_iterations,
            dustbin_score=cfg.word_ot_dustbin_score,
        )
        word_loss, info = contrastive_loss(
            word_matrix, batch["video_id"], batch["text_key"],
            cfg.contrastive_temperature,
        )
        losses["word"] = word_loss
        infos.append(info)

    if cfg.text_representation == "sentence":
        total = losses["sentence"]
    elif cfg.text_representation == "word":
        total = losses["word"]
    else:
        weight_sum = cfg.sentence_loss_weight + cfg.word_loss_weight
        total = (
            cfg.sentence_loss_weight * losses["sentence"]
            + cfg.word_loss_weight * losses["word"]
        ) / weight_sum

    info = {
        key: max(item[key] for item in infos)
        for key in ("active_queries", "active_videos", "masked_pairs")
    }
    return total, info, losses


def run_training(cfg: VideoTextConfig, resume: str | None = None):
    cfg.validate()
    seed_all(cfg.seed)
    splits = check_disjoint_splits(cfg.train_ids, cfg.val_ids, cfg.test_ids)
    dataset = VideoTextDataset(cfg, cfg.train_ids)
    validation = VideoTextDataset(cfg, cfg.val_ids)
    num_videos = len(set(ref.video_id for ref in dataset.refs))
    if num_videos < cfg.batch_size:
        raise ValueError(
            f"Training has {num_videos} source videos, but batch_size={cfg.batch_size}; "
            "reduce batch_size or add videos. Do NOT duplicate same-video queries as negatives."
        )
    if len(set(ref.video_id for ref in validation.refs)) < 2:
        raise ValueError("Validation retrieval needs >= 2 distinct held-out source videos")

    sampler = VideoDiverseBatchSampler(dataset.refs, cfg.batch_size, cfg.seed)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_video_text,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dims = dims(dataset)
    if model_dims != dims(validation):
        raise ValueError("Feature dimensionality differs between train/val caches")
    model = VideoTextModel(cfg, *model_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    mixed = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / cfg.checkpoint_name
    start_epoch, best_r1, global_step = 0, -1.0, 0

    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        if state.get("dims") != list(model_dims):
            raise ValueError("Resume checkpoint feature dimensions differ from caches")
        if checkpoint_architecture(state.get("model_cfg", {})) != architecture_config(cfg):
            raise ValueError("Resume checkpoint model/objective configuration mismatch")
        verify_checkpoint_text_source(state.get("config", {}), cfg)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        global_step = int(state["step"])
        best_r1 = float(state["best_r1"])

    logfile = out_dir / "history.jsonl"
    print(
        f"[video_text] device={device} train={len(dataset)} val={len(validation)} "
        f"videos={num_videos} feature_dims={model_dims} "
        f"representation={cfg.text_representation} split_videos={splits}"
    )
    if num_videos < 8:
        warnings.warn(
            "Few original videos: check that signer and session are separated across splits"
        )
    if not len(loader):
        raise ValueError("No diverse train batches. Check video counts and batch_size")

    stop = False
    for epoch in range(start_epoch, cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running = []
        component_running = {"sentence": [], "word": []}
        timing = time.perf_counter()
        active = 0

        for batch in loader:
            data = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=mixed
            ):
                output = model(data)
                loss, info, component_losses = _training_objective(
                    output, batch, cfg
                )

            if info["active_queries"] == 0 and info["active_videos"] == 0:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite contrastive/partial-OT loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            active += 1
            running.append(float(loss.detach()))
            for key, value in component_losses.items():
                component_running[key].append(float(value.detach()))

            if global_step % 20 == 0:
                elapsed = max(time.perf_counter() - timing, 1e-6)
                print(
                    f"[video_text] epoch={epoch + 1} step={global_step} "
                    f"loss={np.mean(running):.4f} steps/sec={active / elapsed:.2f}"
                )
            if cfg.max_train_steps and global_step >= cfg.max_train_steps:
                stop = True
                break

        if not running:
            raise ValueError(
                "No informative batches: check text duplicates or number of videos"
            )
        valstats = evaluate(model, validation, cfg, device)
        metric = valstats["r1"]
        row = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": float(np.mean(running)),
            "seconds": round(time.perf_counter() - timing, 3),
            "text_representation": cfg.text_representation,
            "val": valstats,
        }
        if component_running["sentence"]:
            row["train_sentence_loss"] = float(np.mean(component_running["sentence"]))
        if component_running["word"]:
            row["train_word_loss"] = float(np.mean(component_running["word"]))
        with logfile.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[video_text] {json.dumps(row, ensure_ascii=False)}")

        if metric is None:
            raise ValueError("Validation has no reliable cross-video distractors")
        if metric > best_r1:
            best_r1 = metric
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "step": global_step,
                    "best_r1": best_r1,
                    "dims": list(model_dims),
                    "model_cfg": architecture_config(cfg),
                    "config": asdict(cfg),
                    "val": valstats,
                },
                checkpoint,
            )
            print(f"[checkpoint] {checkpoint}")
        if stop:
            break
    return checkpoint


def evaluate_checkpoint(cfg: VideoTextConfig, checkpoint_path: str | Path):
    check_disjoint_splits(cfg.train_ids, cfg.val_ids, cfg.test_ids)
    dataset = VideoTextDataset(cfg, cfg.test_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    stored = checkpoint["config"]
    verify_checkpoint_text_source(stored, cfg)
    if checkpoint_architecture(stored) != architecture_config(cfg):
        raise ValueError("Checkpoint model/objective configuration differs from config")
    model = VideoTextModel(cfg, *checkpoint["dims"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return evaluate(model, dataset, cfg, device)
