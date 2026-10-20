
from __future__ import annotations
from persian_sea.core.registry import Registry

text_backends = Registry("text encoder")


def _bge_m3(**kwargs):
    from persian_sea.text.bge_m3 import BGEM3DenseExtractor
    instruction = kwargs.pop("instruction", "")
    if instruction:
        raise ValueError("BGE-M3 backend does not support text_instruction in this pipeline")
    return BGEM3DenseExtractor(**kwargs)


def _qwen3_embedding(**kwargs):
    from persian_sea.text.qwen3_embedding import Qwen3EmbeddingExtractor
    return Qwen3EmbeddingExtractor(**kwargs)


text_backends.register("bge_m3", _bge_m3)
text_backends.register("qwen3_embedding", _qwen3_embedding)


def build_text_encoder(backend: str, **kwargs):
    return text_backends.create(backend, **kwargs)
