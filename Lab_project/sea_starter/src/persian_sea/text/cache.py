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
    def __init__(self, path):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=False)

    @property
    def embeddings(self):
        return self.data["embeddings"]

    @property
    def starts(self):
        return self.data["starts"]

    @property
    def ends(self):
        return self.data["ends"]

    @property
    def texts(self):
        return [str(x) for x in self.data["texts"].tolist()]

    @property
    def word_embeddings(self):
        return self.data["word_embeddings"] if "word_embeddings" in self.data else None

    @property
    def word_cue_offsets(self):
        return self.data["word_cue_offsets"] if "word_cue_offsets" in self.data else None

    @property
    def word_texts(self):
        return [str(x) for x in self.data["word_texts"].tolist()] if "word_texts" in self.data else None

    @property
    def word_char_offsets(self):
        return self.data["word_char_offsets"] if "word_char_offsets" in self.data else None

    def close(self):
        self.data.close()


def text_cache_directory(cfg):
    return cache_directory(cfg.paths.text_feature_cache, cfg.semantic.text_backend,
                           cfg.semantic.text_cache_layout)


def expected_text_identity(cfg):
    pooling = "cls+word_mean_subword" if cfg.semantic.text_backend == "bge_m3" else None
    return identity(
        cfg.semantic.text_backend, cfg.semantic.text_model,
        cfg.semantic.text_instruction, cfg.semantic.text_revision,
        pooling=pooling, max_length=cfg.semantic.text_max_length,
    )


def verify_text_cache(path, cfg, video_id=None):
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
                if ("cache_identity_sha256" not in f or
                    str(f["cache_identity_sha256"].item()) != expected_hash):
                    raise ValueError(f"Text cache NPZ/sidecar identity mismatch: {path}")
            if current is not None and (
                "texts" not in f or [str(s) for s in f["texts"].tolist()] != current
            ):
                raise ValueError(f"Stale text cache text/order: {path}")
            if cfg.semantic.text_backend == "bge_m3":
                required = {"word_embeddings", "word_cue_offsets",
                            "word_texts", "word_char_offsets"}
                missing = required.difference(f.files)
                if missing:
                    raise ValueError(
                        f"Sentence-only BGE cache at {path}; missing {sorted(missing)}. "
                        "Rebuild with overwrite_text_cache=true."
                    )
    if not meta.get("identity"):
        warnings.warn("Legacy BGE cache cannot verify revision/pooling; rebuild it.",
                      stacklevel=2)


def _write_npz_atomic(path, **arrays):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        np.savez_compressed(temp, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def scan_text_video(video_id, cfg, extractor):
    out_dir = text_cache_directory(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz, out_json = out_dir / f"{video_id}.npz", out_dir / f"{video_id}.json"
    if out_npz.exists() and not cfg.semantic.overwrite_text_cache:
        verify_text_cache(out_npz, cfg, video_id)
        print(f"[text] verified & skip existing: {out_npz}")
        return out_npz
    source = find_subtitle(cfg.paths.subtitles_raw, video_id)
    if source is None:
        raise FileNotFoundError(f"No subtitle for {video_id} in {cfg.paths.subtitles_raw}")
    cues = read_subtitles(source)
    texts = [c.text for c in cues]

    word_arrays = {}
    if cfg.semantic.text_backend == "bge_m3":
        batch = extractor.encode_with_words(texts, batch_size=cfg.semantic.text_batch_size)
        embeddings = np.asarray(batch.embeddings)
        word_arrays = {
            "word_embeddings": batch.word_embeddings,
            "word_cue_offsets": batch.cue_offsets,
            "word_texts": batch.word_texts,
            "word_char_offsets": batch.char_offsets,
        }
    else:
        embeddings = np.asarray(extractor.encode(texts, batch_size=cfg.semantic.text_batch_size))

    if embeddings.ndim != 2 or embeddings.shape[0] != len(cues):
        raise ValueError(f"Invalid extracted text features for {video_id}: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Nonfinite extracted text features for {video_id}")
    if word_arrays:
        if not np.isfinite(word_arrays["word_embeddings"]).all():
            raise ValueError(f"Nonfinite BGE word features for {video_id}")
        offsets = word_arrays["word_cue_offsets"]
        if offsets.shape != (len(cues) + 1,) or offsets[0] != 0:
            raise ValueError(f"Invalid word cue offsets for {video_id}")
        if offsets[-1] != len(word_arrays["word_embeddings"]):
            raise ValueError(f"Word cue offsets do not cover words: {video_id}")

    dtype = np.float16 if cfg.semantic.text_cache_dtype == "float16" else np.float32
    metadata = {
        "video_id": video_id, "subtitle": str(source),
        "text_model": cfg.semantic.text_model, "num_cues": len(cues),
        "hidden_size": int(embeddings.shape[-1]),
        "num_words": int(len(word_arrays.get("word_texts", []))),
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
        arrays = {
            "embeddings": embeddings.astype(dtype),
            "starts": np.asarray([c.start for c in cues], np.float32),
            "ends": np.asarray([c.end for c in cues], np.float32),
            "texts": np.asarray(texts),
            "hidden": np.asarray([embeddings.shape[-1]], np.int32),
            "cache_identity_sha256": np.asarray(identity_digest(metadata["identity"])),
        }
        for key, value in word_arrays.items():
            arrays[key] = value.astype(dtype) if key == "word_embeddings" else value
        _write_npz_atomic(out_npz, **arrays)
    finally:
        tmp_meta.unlink(missing_ok=True)
    print(f"[text] {video_id}: {len(cues)} cues, embeddings={embeddings.shape}, "
          f"words={len(word_arrays.get('word_texts', []))}, backend={cfg.semantic.text_backend}")
    return out_npz


def scan_text_all(cfg):
    extractor = build_text_encoder(
        cfg.semantic.text_backend, model_name=cfg.semantic.text_model,
        device=cfg.semantic.text_device, amp=cfg.semantic.text_amp,
        max_length=cfg.semantic.text_max_length,
        instruction=cfg.semantic.text_instruction,
        revision=cfg.semantic.text_revision,
    )
    for video_id in read_video_ids(cfg.paths.video_ids):
        scan_text_video(video_id, cfg, extractor)
