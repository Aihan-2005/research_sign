
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

NORMALIZATION_VERSION = "fa-subtitle-v1"


def safe_backend_name(backend: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", backend):
        raise ValueError(f"Invalid text backend name: {backend!r}")
    return backend


def cache_directory(root: str | Path, backend: str, layout: str) -> Path:
    root = Path(root)
    if layout == "legacy":
        return root
    if layout == "by_backend":
        return root / safe_backend_name(backend)
    raise ValueError("text_cache_layout must be 'legacy' or 'by_backend'")


def identity_digest(meta: dict) -> str:
    """Stable signature recorded in both cache NPZ and JSON sidecar."""
    payload = json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_digest(texts: list[str]) -> str:
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def identity(backend: str, model_name: str, instruction: str = "",
             revision: str = "main", *, pooling: str | None = None,
             max_length: int = 256) -> dict:
    safe_backend_name(backend)
    if backend not in ("bge_m3", "qwen3_embedding"):
        raise ValueError(f"Unsupported text backend: {backend}")
    pooling = pooling or ("cls" if backend == "bge_m3" else "last_token")
    return {
        "schema_version": 2,
        "backend": backend,
        "model_name": model_name,
        "model_revision": revision,
        "pooling": pooling,
        "instruction": instruction,
        "normalization_version": NORMALIZATION_VERSION,
        "max_length": int(max_length),
    }


def validate_sidecar(npz_path: str | Path, *, expected: dict,
                     strict: bool = True, texts: list[str] | None = None) -> dict:

    path = Path(npz_path).with_suffix(".json")
    if not path.is_file():
        if strict:
            raise ValueError(f"Text cache metadata missing: {path}. Rescan text.")
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid text cache metadata: {path}") from exc
    meta = raw.get("identity")
    if not isinstance(meta, dict):
        if strict:
            raise ValueError(f"Legacy text cache cannot be verified: {path}; rebuild it")
        if expected.get("backend") not in (None, "", "bge_m3"):
            raise ValueError(f"Legacy cache cannot represent {expected['backend']}")
        if expected.get("instruction"):
            raise ValueError("Instruction-aware cache must be regenerated")
        expected_model = expected.get("model_name")
        if expected_model and raw.get("text_model") != expected_model:
            raise ValueError(f"Wrong cached model in {path}: {raw.get('text_model')!r}")
        return raw
    for key, value in expected.items():
        if value is not None and meta.get(key) != value:
            raise ValueError(f"Stale text cache {path}: {key}={meta.get(key)!r}, expected={value!r}")
    if texts is not None and raw.get("text_sha256") != text_digest(texts):
        raise ValueError(f"Stale text cache {path}: subtitle content hash mismatch")
    return raw
