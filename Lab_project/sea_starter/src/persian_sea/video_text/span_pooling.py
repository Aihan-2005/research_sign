
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class SpanPooler(nn.Module):
    def __init__(self, dim: int, sizes: tuple[int, ...], backend: str = "mean"):
        super().__init__()
        if backend not in {"mean", "attentive"}:
            raise ValueError("span backend must be 'mean' or 'attentive'")
        if dim < 1 or not sizes or any(n < 1 for n in sizes):
            raise ValueError("Invalid span sizes or dimension")
        self.sizes, self.backend = tuple(sizes), backend
        if backend == "attentive":
            self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                       nn.Tanh(), nn.Linear(dim, 1))
            self.relative_position = nn.Linear(1, dim, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("Expected tokens [B,M,D] and mask [B,M]")
        if not mask.any(dim=1).all():
            raise ValueError("Each example needs valid visual segments")
        B, M, D = tokens.shape
        vectors, valid = [], []
        for length in self.sizes:
            if length > M:
                continue
            # [B, M-length+1, length, D]
            chunks = tokens.unfold(1, length, 1).permute(0, 1, 3, 2)
            usable = mask.unfold(1, length, 1).all(dim=-1)
            if self.backend == "mean":
                pooled = chunks.mean(dim=2)
            else:
                positions = torch.linspace(-1, 1, length, device=tokens.device,
                                           dtype=tokens.dtype).view(1, 1, length, 1)
                contextual = chunks + self.relative_position(positions)
                weights = torch.softmax(self.score(contextual).squeeze(-1), dim=2)
                pooled = (weights[..., None] * contextual).sum(dim=2)
            vectors.append(F.normalize(pooled, dim=-1))
            valid.append(usable)
        if not vectors:
            raise ValueError("No spans fit candidate length")
        all_vectors = torch.cat(vectors, dim=1)
        all_valid = torch.cat(valid, dim=1)
        if not all_valid.any(dim=1).all():
            raise ValueError("Example without a complete valid span")
        # Zero padded output; invalid spans are independently masked in loss.
        return all_vectors * all_valid[..., None], all_valid
