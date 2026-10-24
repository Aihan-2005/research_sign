
from __future__ import annotations

from dataclasses import asdict
from functools import partial
from pathlib import Path
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.data import TranslationCueDataset, collate
from persian_sea.translation.model import build_model
from persian_sea.translation.readiness import readiness_report, write_report


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_tensors(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def save_atomic(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@torch.no_grad()
def validation_loss(model, loader, device) -> float:
    model.eval()
    total = 0.0
    items = 0
    for batch in loader:
        batch = move_tensors(batch, device)
        loss = model(batch).loss.float()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        n = len(batch["text"])
        total += float(loss) * n
        items += n
    if not items:
        raise ValueError("Empty validation set")
    return total / items


def run_training(cfg: TranslationConfig, resume: str | None = None) -> Path:
    cfg.validate()
    # Do not download a 1GB seq2seq decoder before the data checks finish.
    report = readiness_report(cfg, require_all=True)
    write_report(report, Path(cfg.output_dir) / "readiness.json")
    if not report["ok"]:
        raise ValueError("Data not ready; see readiness.json: " + "; ".join(report["errors"][:6]))

    device = device_auto()
    if device.type != "cuda":
        raise RuntimeError("Real translation training requires CUDA in this first release; "
                           "run check on Mac and train on the university GPU server")
    from transformers import AutoTokenizer
    seed_everything(cfg.seed)
    train_set = TranslationCueDataset(cfg, cfg.train_ids)
    val_set = TranslationCueDataset(cfg, cfg.val_ids)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    make_batch = partial(collate, tokenizer=tokenizer,
                         max_target_tokens=cfg.max_target_tokens)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, collate_fn=make_batch,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, collate_fn=make_batch)
    sample = train_set[0]
    rgb_dim, views = int(sample["rgb"].shape[-1]), int(sample["rgb"].shape[-2])
    model = build_model(cfg, rgb_dim=rgb_dim, views=views).to(device)
    optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    best = out_dir / "best_translation.pt"
    latest = out_dir / "last_translation.pt"
    start_epoch, step, best_loss = 0, 0, float("inf")
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        if ckpt["config"] != asdict(cfg) or tuple(ckpt["rgb_dims"]) != (rgb_dim, views):
            raise ValueError("Resume config/cache dimensions changed")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"])
        step = int(ckpt["step"])
        best_loss = float(ckpt["best_val_loss"])
    print(f"[translation] device={device}; train_cues={len(train_set)}; "
          f"validation_cues={len(val_set)}; model={cfg.model_name}; "
          f"visual_frozen={cfg.freeze_visual}", flush=True)
    if step >= cfg.max_train_steps > 0:
        raise ValueError("Resume checkpoint already reached max_train_steps")
    stop = False
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0
        begin = time.monotonic()
        train_losses = []
        for batch_index, batch in enumerate(train_loader):
            batch = move_tensors(batch, device)
            result = model(batch)
            loss = result.loss.float()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite train loss; stop and inspect inputs / learning rate")

            is_end = batch_index + 1 == len(train_loader)
            (loss / cfg.accumulation_steps).backward()
            accumulation += 1
            train_losses.append(float(loss.detach()))
            if accumulation == cfg.accumulation_steps or is_end:
                if accumulation != cfg.accumulation_steps:
                    scale = cfg.accumulation_steps / accumulation
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0
                step += 1
                if step % 10 == 0:
                    print(f"[translation] epoch={epoch + 1} step={step} "
                          f"loss={sum(train_losses)/len(train_losses):.4f} "
                          f"elapsed={time.monotonic()-begin:.1f}s", flush=True)
                if cfg.max_train_steps and step >= cfg.max_train_steps:
                    stop = True
                    break
        if not train_losses:
            raise ValueError("No translation train minibatches")
        val = validation_loss(model, val_loader, device)
        row = {"epoch": epoch + 1, "step": step, "train_loss": float(np.mean(train_losses)),
               "val_loss": val, "seconds": round(time.monotonic() - begin, 1)}
        with (out_dir / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[translation] " + json.dumps(row), flush=True)
        improved = val < best_loss
        if improved:
            best_loss = val
        payload = {"schema_version": 1, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "config": asdict(cfg),
                   "rgb_dims": [rgb_dim, views], "epoch": epoch + 1,
                   "step": step, "best_val_loss": best_loss}
        save_atomic(payload, latest)
        if improved:

            best_payload = dict(payload)
            best_payload.pop("optimizer")
            save_atomic(best_payload, best)
            print("[translation] best:", best, flush=True)
        if stop:
            break
    if not best.is_file():
        raise RuntimeError("No best translation checkpoint saved")
    return best
