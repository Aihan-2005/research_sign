

from __future__ import annotations

from pathlib import Path
from persian_sea.translation.config import TranslationConfig


def load_tokenizer(cfg: TranslationConfig, checkpoint_dir: str | Path | None = None):
    from transformers import AutoTokenizer
    path = Path(checkpoint_dir) / "tokenizer" if checkpoint_dir else None
    source = str(path) if path is not None and path.is_dir() else cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
    if cfg.translation_backend == "mbart50":
        if cfg.target_lang not in tokenizer.lang_code_to_id:
            raise ValueError(f"Tokenizer does not support language {cfg.target_lang}")
        tokenizer.src_lang = cfg.target_lang
        tokenizer.tgt_lang = cfg.target_lang
    return tokenizer


def configure_generation(model, tokenizer, cfg: TranslationConfig):
    if cfg.translation_backend == "mbart50":
        bos = tokenizer.lang_code_to_id[cfg.target_lang]
        model.generation_config.forced_bos_token_id = bos
    return model
