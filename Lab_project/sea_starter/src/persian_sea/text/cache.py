
from __future__ import annotations
from pathlib import Path
import json
import os
import tempfile
import warnings
import numpy as np

from persian_sea.config import AppConfig, read_video_ids
from persian_sea.io.subtitles import find_subtitle, read_subtitles
from persian_sea.text.backends import build_text_encoder
from persian_sea.text.cache_identity import (
    identity, identity_digest, cache_directory, text_digest, validate_sidecar,
)


class TextFeatureCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=False)

    @property
    def embeddings(self) -> np.ndarray:
        return self.data["embeddings"]

    @property
    def starts(self) -> np.ndarray:
        return self.data["starts"]

    @property
    def ends(self) -> np.ndarray:
        return self.data["ends"]

    @property
    def texts(self) -> list[str]:
        return [str(x) for x in self.data["texts"].tolist()]

    def close(self) -> None:
        self.data.close()


def text_cache_directory(cfg: AppConfig) -> Path:
    return cache_directory(cfg.paths.text_feature_cache,
                           cfg.semantic.text_backend, cfg.semantic.text_cache_layout)


def expected_text_identity(cfg: AppConfig) -> dict:
    return identity(cfg.semantic.text_backend, cfg.semantic.text_model,
                    cfg.semantic.text_instruction, cfg.semantic.text_revision,
                    max_length=cfg.semantic.text_max_length)


def verify_text_cache(path: Path, cfg: AppConfig,
                      video_id: str | None = None) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    expected = expected_text_identity(cfg)
    strict = cfg.semantic.text_cache_layout != "legacy" or cfg.semantic.text_backend != "bge_m3"
    current = None
    if video_id is not None:
        subtitle = find_subtitle(cfg.paths.subtitles_raw, video_id)
        if subtitle is None:
            raise FileNotFoundError(f"No subtitle for {video_id}")
        current = [c.text for c in read_subtitles(subtitle)]
    meta = validate_sidecar(path, expected=expected, strict=strict,
                            texts=current if strict else None)
    if current is not None or meta.get("identity"):
        with np.load(path, allow_pickle=False) as f:
            if meta.get("identity"):
                expected_hash = identity_digest(meta["identity"])
                if "cache_identity_sha256" not in f or str(f["cache_identity_sha256"].item()) != expected_hash:
                    raise ValueError(f"Text cache NPZ/sidecar identity mismatch: {path}")
            if current is not None and ("texts" not in f or [str(s) for s in f["texts"].tolist()] != current):
                raise ValueError(f"Stale text cache text/order: {path}")
    if not meta.get("identity"):
        warnings.warn("Legacy BGE-M3 cache: revision, pooling and normalization cannot "
                      "be verified. Rescan with text_cache_layout=by_backend for rigorous ablations.",
                      stacklevel=2)


def _write_npz_atomic(path: Path, **arrays) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        np.savez_compressed(temp, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def scan_text_video(video_id: str, cfg: AppConfig, extractor) -> Path:
    out_dir = text_cache_directory(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{video_id}.npz"
    out_json = out_dir / f"{video_id}.json"
    if out_npz.exists() and not cfg.semantic.overwrite_text_cache:
        verify_text_cache(out_npz, cfg, video_id)
        print(f"[text] verified & skip existing: {out_npz}")
        return out_npz
    source = find_subtitle(cfg.paths.subtitles_raw, video_id)
    if source is None:
        raise FileNotFoundError(f"No subtitle for {video_id} in {cfg.paths.subtitles_raw}")
    cues = read_subtitles(source)
    texts = [c.text for c in cues]
    embeddings = np.asarray(extractor.encode(texts, batch_size=cfg.semantic.text_batch_size))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(cues):
        raise ValueError(f"Invalid extracted text features for {video_id}: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Nonfinite extracted text features for {video_id}")
    dtype = np.float16 if cfg.semantic.text_cache_dtype == "float16" else np.float32
    metadata = {
        "video_id": video_id, "subtitle": str(source),
        "text_model": cfg.semantic.text_model,
        "num_cues": len(cues), "hidden_size": int(embeddings.shape[-1]),
        "identity": expected_text_identity(cfg),
        "text_sha256": text_digest(texts),
        "resolved_model_revision": getattr(extractor, "resolved_revision", None),
    }
    
    with tempfile.NamedTemporaryFile(mode="w", dir=out_dir, suffix=".json",
                                     encoding="utf-8", delete=False) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        tmp_meta = Path(f.name)
    try:
        os.replace(tmp_meta, out_json)
        _write_npz_atomic(
            out_npz,
            embeddings=embeddings.astype(dtype),
            starts=np.asarray([c.start for c in cues], dtype=np.float32),
            ends=np.asarray([c.end for c in cues], dtype=np.float32),
            texts=np.asarray(texts),
            hidden=np.asarray([embeddings.shape[-1]], dtype=np.int32),
            cache_identity_sha256=np.asarray(identity_digest(metadata["identity"])),
        )
    finally:
        tmp_meta.unlink(missing_ok=True)
    print(f"[text] {video_id}: {len(cues)} cues, embeddings={embeddings.shape}, backend={cfg.semantic.text_backend}")
    return out_npz


def scan_text_all(cfg: AppConfig) -> None:
    extractor = build_text_encoder(
        cfg.semantic.text_backend,
        model_name=cfg.semantic.text_model,
        device=cfg.semantic.text_device,
        amp=cfg.semantic.text_amp,
        max_length=cfg.semantic.text_max_length,
        instruction=cfg.semantic.text_instruction,
        revision=cfg.semantic.text_revision,
    )
    for video_id in read_video_ids(cfg.paths.video_ids):
        scan_text_video(video_id, cfg, extractor)
