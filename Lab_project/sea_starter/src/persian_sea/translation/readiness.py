from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from persian_sea.io.subtitles import find_subtitle, read_subtitles
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
            errors.append("Need at least two independent TRAIN videos for VLP warm-up")
        if counts["val"] < 1 or counts["test"] < 1:
            errors.append("Need separate validation and test recordings")
        if cfg.signer_metadata:
            from persian_sea.video_text.split_audit import audit_signer_splits
            result["signer_audit"] = audit_signer_splits(
                cfg.signer_metadata, cfg.train_ids, cfg.val_ids, cfg.test_ids,
                strict_signer=cfg.strict_signer)
            if result["signer_audit"].get("signers_across_splits"):
                result["warnings"].append("Signers appear across multiple splits; signer-independent quality may be optimistic")
            if result["signer_audit"].get("unknown_metadata"):
                result["warnings"].append("Signer/session metadata incomplete")
        else:
            result["warnings"].append(
                "No signer/source_recording_id metadata: cannot verify signer/session leakage")
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
                errors.append(f"{split}/{vid}: missing VTT or Visual NPZ: {sub}, {path}")
                continue
            try:
                texts = read_subtitles(sub)
                if not texts or not any(c.text.strip() for c in texts):
                    raise ValueError("empty text cues")
                with np.load(path, allow_pickle=False) as f:
                    # Full tensor integrity check; expensive only for large caches.
                    d = {k: f[k] for k in ("pose", "rgb", "starts", "ends")}
                validate_cache(d, path)
                signature = (tuple(d["pose"].shape[1:]), tuple(d["rgb"].shape[1:]))
                if feature_signature is None:
                    feature_signature = signature
                elif signature != feature_signature:
                    raise ValueError(f"Visual dimensions changed across videos: {signature} != {feature_signature}")
                videos += 1
                cues += sum(bool(c.text.strip()) for c in texts)
            except (ValueError, OSError, KeyError) as exc:
                errors.append(f"{split}/{vid}: {exc}")
        checked[split] = {"videos": videos, "nonempty_cues": cues}
    result["checked"] = checked
    if feature_signature is not None:
        result["visual_shapes_without_video_axis"] = [list(s) for s in feature_signature]
    if cfg.require_pretrained_visual and not result["visual_pretrained"]:
        if require_all:
            errors.append(f"VLP checkpoint missing: {cfg.visual_checkpoint}")
        else:
            result["warnings"].append("VLP checkpoint not available yet: translation training blocked")
    result["ok"] = not errors
    return result


def write_report(result: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
