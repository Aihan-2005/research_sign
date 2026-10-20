
from __future__ import annotations
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden [B,L,D] and attention_mask [B,L]")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    indices = positions[None, :].masked_fill(attention_mask == 0, -1).amax(dim=1)
    if (indices < 0).any():
        raise ValueError("Cannot pool a fully padded sequence")
    return hidden[torch.arange(hidden.size(0), device=hidden.device), indices]


class Qwen3EmbeddingExtractor:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 device: str = "auto", amp: bool = True,
                 max_length: int = 256, instruction: str = "",
                 revision: str = "main"):
        from transformers import AutoModel, AutoTokenizer
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.max_length = int(max_length)
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        if not texts:
            return np.empty((0, self.hidden_size), dtype=np.float32)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        examples = ([f"Instruct: {self.instruction}\nQuery: {t}" for t in texts]
                    if self.instruction else texts)
        result = []
        for start in range(0, len(examples), batch_size):
            batch = self.tokenizer(examples[start:start+batch_size],
                                   padding=True, truncation=True,
                                   max_length=self.max_length,
                                   return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            context = (torch.autocast("cuda", dtype=torch.float16)
                       if self.amp else nullcontext())
            with context:
                hidden = self.model(**batch).last_hidden_state
                pooled = last_token_pool(hidden, batch["attention_mask"])
                pooled = F.normalize(pooled.float(), p=2, dim=-1)
            result.append(pooled.cpu().numpy())
        return np.concatenate(result).astype(np.float32)
