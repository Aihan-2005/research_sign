
from __future__ import annotations
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BGEM3DenseExtractor:
    def __init__(self, model_name: str = "BAAI/bge-m3",
                 device: str = "auto", amp: bool = True,
                 max_length: int = 256, revision: str = "main"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision)
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.empty((0, self.hidden_size), dtype=np.float32)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        chunks = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[start:start+batch_size],
                                    padding=True, truncation=True,
                                    max_length=self.max_length,
                                    return_tensors="pt")
            tokens = {key: value.to(self.device) for key,value in tokens.items()}
            context = (torch.autocast("cuda", dtype=torch.float16)
                       if self.amp else nullcontext())
            with context:
                hidden = self.model(**tokens).last_hidden_state[:, 0]
                dense = F.normalize(hidden.float(), p=2, dim=-1)
            chunks.append(dense.cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)
