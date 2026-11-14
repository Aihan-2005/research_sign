from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import VideoTextConfig
from .span_pooling import SpanPooler


class VideoTower(nn.Module):
    def __init__(self, visual: nn.Module, dim: int, layers: int,
                 heads: int, microbatch: int):
        super().__init__()
        self.visual = visual
        self.microbatch = int(microbatch)
        if layers:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 2,
                batch_first=True, norm_first=True, dropout=.1, activation="gelu"
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            self.temporal = None
        self.time_projection = nn.Sequential(
            nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, pose, rgb, starts, ends, mask):
        batch_size, max_segments = mask.shape
        flat_idx = torch.where(mask.reshape(-1))[0]
        if flat_idx.numel() == 0:
            raise ValueError("Every batch needs at least one sign segment")
        flat_pose = pose.reshape(batch_size * max_segments, *pose.shape[2:])
        flat_rgb = rgb.reshape(batch_size * max_segments, *rgb.shape[2:])
        groups = []
        for ids in flat_idx.split(self.microbatch):
            groups.append(self.visual(flat_pose[ids], flat_rgb[ids])["fused"])
        encoded = torch.cat(groups, dim=0)
        if encoded.shape[-1] != self.dim:
            raise ValueError(f"Visual output dimension must equal {self.dim}")
        flat = encoded.new_zeros(batch_size * max_segments, self.dim)
        flat = flat.index_copy(0, flat_idx, encoded)
        tokens = flat.reshape(batch_size, max_segments, self.dim)

        base = starts.masked_fill(~mask, float("inf")).amin(dim=1, keepdim=True)
        limit = ends.masked_fill(~mask, -float("inf")).amax(dim=1, keepdim=True)
        extent = (limit - base).clamp_min(1e-3)
        middle = (starts + ends) / 2
        rel_middle = 2 * (middle - base) / extent - 1
        length = (ends - starts).clamp_min(1e-3)
        prev = F.pad(ends[:, :-1], (1, 0), value=0)
        prev[:, 0] = starts[:, 0]
        gap = (starts - prev).clamp_min(0)
        timing = torch.stack(
            [rel_middle, torch.log1p(length), torch.log1p(gap)], dim=-1
        )
        timing = timing.masked_fill(~mask[..., None], 0.0)
        tokens = tokens + self.time_projection(timing)
        if self.temporal is not None:
            tokens = self.temporal(tokens, src_key_padding_mask=~mask)
        tokens = self.layer_norm(tokens)
        return F.normalize(tokens, dim=-1) * mask[..., None]


class TextTower(nn.Module):
    def __init__(self, text_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.network(x.float()), dim=-1)


def multi_scale_spans(tokens, mask, sizes: tuple[int, ...]):

    batch_size, max_segments, dim = tokens.shape
    sums = torch.cat(
        (tokens.new_zeros((batch_size, 1, dim)), tokens.cumsum(dim=1)), dim=1
    )
    counts = torch.cat(
        (tokens.new_zeros((batch_size, 1, 1)),
         mask.to(tokens.dtype).cumsum(dim=1)[..., None]), dim=1
    )
    spans, valid = [], []
    for length in sizes:
        if length > max_segments:
            continue
        pooled = (sums[:, length:] - sums[:, :-length]) / length
        usable = (counts[:, length:] - counts[:, :-length]).squeeze(-1) == length
        spans.append(F.normalize(pooled, dim=-1))
        valid.append(usable)
    if not spans:
        raise ValueError("No valid span sizes")
    combined = torch.cat(spans, dim=1)
    mask_out = torch.cat(valid, dim=1)
    if not mask_out.any(dim=1).all():
        raise ValueError("Some video bags contain no valid spans")
    return combined, mask_out


class VideoTextModel(nn.Module):
    def __init__(self, cfg: VideoTextConfig, text_dim: int, rgb_dim: int,
                 num_views: int, visual: nn.Module | None = None):
        super().__init__()
        if visual is None:
            from persian_sea.model.builders import build_visual_encoder
            visual = build_visual_encoder(
                cfg.visual_backend, pose_input_dim=4, rgb_input_dim=rgb_dim,
                num_views=num_views, model_dim=cfg.model_dim,
                output_dim=cfg.embedding_dim,
            )
        self.video = VideoTower(
            visual, cfg.embedding_dim, cfg.temporal_layers,
            cfg.temporal_heads, cfg.visual_microbatch,
        )


        self.text = TextTower(text_dim, cfg.embedding_dim)
        self.word_text = (
            TextTower(text_dim, cfg.embedding_dim)
            if cfg.text_representation in {"word", "both"} else None
        )
        self.text_representation = cfg.text_representation
        self.span_sizes = cfg.span_sizes
        self.span_pooler = SpanPooler(
            cfg.embedding_dim, cfg.span_sizes, backend=cfg.span_pooling
        )
        self.mil_temperature = cfg.mil_temperature

    def encode_video(self, batch: dict):

        return self.video(
            batch["pose"], batch["rgb"], batch["starts"],
            batch["ends"], batch["mask"],
        )

    def forward(self, batch: dict):
        tokens = self.encode_video(batch)
        spans, span_mask = self.span_pooler(tokens, batch["mask"])
        result = {
            "video_tokens": tokens,
            "span_vectors": spans,
            "span_mask": span_mask,
        }
        if self.text_representation in {"sentence", "both"}:
            result["text_vectors"] = self.text(batch["text_embedding"])
        if self.text_representation in {"word", "both"}:
            if self.word_text is None or "word_embeddings" not in batch:
                raise ValueError("Word-level mode needs word_embeddings in the batch")
            word_vectors = self.word_text(batch["word_embeddings"])
            result["word_vectors"] = word_vectors * batch["word_mask"][..., None]
            result["word_mask"] = batch["word_mask"]
        return result


def mil_similarity_matrix(text_vectors, span_vectors, span_mask, temperature: float):

    temp = max(float(temperature), 1e-4)
    sims = torch.einsum(
        "id,jsd->ijs", text_vectors.float(), span_vectors.float()
    )
    sims = sims.masked_fill(~span_mask[None, :, :], -1e4)
    count = span_mask.sum(dim=1).clamp_min(1).float().log()
    return temp * torch.logsumexp(sims / temp, dim=2) - temp * count[None, :]
