from __future__ import annotations

"""Fail-closed readiness checks for weakly supervised translation training."""

from pathlib import Path
import json
import numpy as np

from persian_sea.io.subtitles import find_subtitle, read_subtitles
from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
from persian_sea.video_text.dataset import check_disjoint_splits
from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.data import load_ids, validate_cache


def readiness_report(cfg: TranslationConfig, *, require_all: bool = True) -> dict:
    cfg.validate()
    result = {"ok": False, "errors": [], "warnings": [], "splits": {},
              "visual_pretrained": Path(cfg.visual_checkpoint).is_file()}
    errors = result["errors"]
    try:
        counts = check_disjoint_splits(cfg.train_ids, cfg.val_ids, cfg.test_ids)
        result["splits"] = counts
        if counts["train"] < 2:
            errors.append("Training needs at least two distinct videos")
        if counts["val"] < 2:
            errors.append("Validation must have >=2 distinct videos (also needed for VLP retrieval)")
        if counts["test"] < 1:
            errors.append("Test must be a separate nonempty set of videos")
        if cfg.signer_metadata:
            from persian_sea.video_text.split_audit import audit_signer_splits
            result["signer_audit"] = audit_signer_splits(
                cfg.signer_metadata, cfg.train_ids, cfg.val_ids, cfg.test_ids,
                strict_signer=cfg.strict_signer,
            )
            if result["signer_audit"].get("signers_across_splits"):
                result["warnings"].append(
                    "Signers overlap across splits; signer-independent generalization is not established"
                )
            if result["signer_audit"].get("unknown_metadata"):
                result["warnings"].append("Signer/session metadata incomplete")
        else:
            if cfg.strict_signer:
                errors.append("strict_signer=True requires signer_metadata CSV")
            result["warnings"].append(
                "No signer/source recording metadata: source/signer leakage cannot be audited"
            )
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"Split/metadata audit: {exc}")
        return result

    checked = {}
    feature_signature = None
    for split, ids_path in (("train", cfg.train_ids), ("val", cfg.val_ids),
                            ("test", cfg.test_ids)):
        videos = 0
        cues = 0
        for vid in load_ids(ids_path):
            sub = find_subtitle(Path(cfg.subtitles_dir), vid)
            path = Path(cfg.visual_cache_dir) / f"{vid}.npz"
            if sub is None or not path.is_file():
                errors.append(f"{split}/{vid}: missing VTT or visual NPZ: {sub}, {path}")
                continue
            try:
                texts = read_subtitles(sub)
                if not texts or not any(c.text.strip() for c in texts):
                    raise ValueError("empty text cues")

                # IMPORTANT: propagate require_alignment_review here as well as
                # in the Dataset, so 'check' cannot falsely pass unsafe pairs.
                load_alignment_cues(vid, texts, AlignmentSourceSettings(
                    mode=cfg.pair_source,
                    approved_dir=cfg.approved_subtitles_dir,
                    sea_dir=cfg.sea_alignment_dir,
                    semantic_dir=cfg.semantic_alignment_dir,
                    max_shift_seconds=cfg.max_alignment_shift_seconds,
                    require_alignment_review=cfg.require_alignment_review,
                ))
                with np.load(path, allow_pickle=False) as archive:
                    values = {k: archive[k] for k in ("pose", "rgb", "starts", "ends")}
                validate_cache(values, path)
                signature = (tuple(values["pose"].shape[1:]),
                             tuple(values["rgb"].shape[1:]))
                if feature_signature is None:
                    feature_signature = signature
                elif signature != feature_signature:
                    raise ValueError(
                        f"Visual dimensions differ between videos: {signature} vs {feature_signature}"
                    )
                videos += 1
                cues += sum(bool(c.text.strip()) for c in texts)
            except (ValueError, OSError, KeyError) as exc:
                errors.append(f"{split}/{vid}: {exc}")
        checked[split] = {"videos": videos, "nonempty_cues": cues,
                          "pair_source": cfg.pair_source}
    result["checked"] = checked
    if feature_signature is not None:
        result["visual_shapes_without_video_axis"] = [list(s) for s in feature_signature]
    if cfg.require_pretrained_visual and not result["visual_pretrained"]:
        if require_all:
            errors.append(f"VLP checkpoint missing: {cfg.visual_checkpoint}")
        else:
            result["warnings"].append("VLP checkpoint is not available; training blocked")
    result["ok"] = not errors
    return result


def write_report(result: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
