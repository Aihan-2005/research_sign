

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.data import TranslationCueDataset, collate
from persian_sea.translation.model import build_model
from persian_sea.translation.train import device_auto, move_tensors


@torch.no_grad()
def evaluate_split(cfg: TranslationConfig, split: str, checkpoint: str,
                   output: str | None = None) -> Path:
    if split not in ("val", "test"):
        raise ValueError("Only disjoint validation and test splits can be evaluated")
    from persian_sea.translation.language import load_tokenizer
    import sacrebleu

    ids = cfg.val_ids if split == "val" else cfg.test_ids
    dataset = TranslationCueDataset(cfg, ids)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trained_cfg = TranslationConfig(**payload["config"]).validate()
    if cfg != trained_cfg:
        raise ValueError("Evaluation config differs from the checkpoint config")
    tokenizer_dir = Path(checkpoint).parent / "tokenizer"
    tokenizer = load_tokenizer(cfg, checkpoint_dir=Path(checkpoint).parent)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate, tokenizer=tokenizer,
                                           max_target_tokens=cfg.max_target_tokens,
                                           translation_backend=cfg.translation_backend))
    item = dataset[0]
    dims = (item["rgb"].shape[-1], item["rgb"].shape[-2])
    if tuple(payload["rgb_dims"]) != dims:
        raise ValueError("Feature cache dims differ from checkpoint")
    model = build_model(replace(cfg, require_pretrained_visual=False), *dims)
    model.load_state_dict(payload["model"], strict=True)
    device = device_auto()
    model.to(device).eval()
    predictions, targets, rows = [], [], []
    loss_total, n_total = 0., 0
    for batch in loader:
        moved = move_tensors(batch, device)
        loss = float(model(moved).loss.float())
        generated = model.generate(moved, max_new_tokens=cfg.max_generate_tokens)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for ref, pred in zip(batch["text"], decoded):
            targets.append(ref)
            predictions.append(pred.strip())
            rows.append({"reference": ref, "prediction": pred.strip()})
        n_total += len(batch["text"])
        loss_total += loss * len(batch["text"])
    if not predictions:
        raise ValueError("No evaluated samples")
    report = {
        "split": split, "evaluation_protocol": "weak_vtt_guided_cue_windows_not_vtt_free",
        "checkpoint": str(checkpoint), "num_cues": len(predictions),
        "teacher_forced_loss": loss_total / n_total,
        "corpus_bleu": sacrebleu.corpus_bleu(predictions, [targets]).score,
        "corpus_chrf": sacrebleu.corpus_chrf(predictions, [targets]).score,
        "pairs": rows,
        "warning": "Scores use VTT to SELECT visual windows. They are NOT subtitle-free continuous SLT scores.",
    }
    path = Path(output) if output else Path(cfg.output_dir) / f"eval_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
