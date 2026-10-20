from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn

from persian_sea.model.builders import build_visual_encoder
from persian_sea.semantic.candidates import CandidateSpan, enumerate_candidate_spans


class CachedTextProjector(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), dim=-1)


class CandidateContextEncoder(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 3, heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, segment_embeddings: torch.Tensor,
                starts: torch.Tensor, ends: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if not mask.any(dim=1).all():
            raise ValueError("Every example needs >= 1 valid candidate")
        valid_start = starts.masked_fill(~mask, torch.inf).amin(dim=1, keepdim=True)
        valid_end = ends.masked_fill(~mask, -torch.inf).amax(dim=1, keepdim=True)
        total_duration = (valid_end - valid_start).clamp_min(1e-3)
        middle = (starts + ends) * 0.5
        relative = 2.0 * (middle - valid_start) / total_duration - 1.0
        length = (ends - starts).clamp_min(1e-3)
        previous_end = F.pad(ends[:, :-1], (1, 0), value=0.0)
        previous_end[:, :1] = starts[:, :1]
        gap = (starts - previous_end).clamp_min(0.0)
        time_features = torch.stack([relative, torch.log1p(length), torch.log1p(gap)], dim=-1)
        time_features = time_features.masked_fill(~mask[..., None], 0.0)
        h = segment_embeddings + self.time_mlp(time_features)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.final_norm(h) * mask.unsqueeze(-1).to(h.dtype)


@dataclass
class SampleSpans:
    spans: list[CandidateSpan]
    vectors: torch.Tensor
    semantic_scores: torch.Tensor
    final_scores: torch.Tensor


class SemanticAligner(nn.Module):
    def __init__(self, text_input_dim: int = 1024, rgb_input_dim: int = 768,
                 num_views: int = 4, embedding_dim: int = 512,
                 visual_backend: str = "pose_rgb"):
        super().__init__()
        self.visual = build_visual_encoder(visual_backend, 
            pose_input_dim=4, rgb_input_dim=rgb_input_dim,
            model_dim=384, output_dim=embedding_dim, num_views=num_views,
        )
        self.text = CachedTextProjector(text_input_dim, embedding_dim)
        self.context = CandidateContextEncoder(dim=embedding_dim)
        self.span_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim),
        )

    def encode_segments(self, pose: torch.Tensor, rgb: torch.Tensor,
                        candidate_mask: torch.Tensor) -> torch.Tensor:

        B, M = pose.shape[:2]
        if pose.shape[-1] != 4:
            raise ValueError("Expected cached MediaPipe pose with confidence: (...,75,4)")
        valid = candidate_mask.reshape(-1)
        flat_pose = pose.reshape(B * M, *pose.shape[2:])
        flat_rgb = rgb.reshape(B * M, *rgb.shape[2:])
        embedding_dim = self.text.net[-1].out_features
        output = pose.new_zeros((B * M, embedding_dim), dtype=torch.float32)
        if valid.any():
            output[valid] = self.visual(flat_pose[valid], flat_rgb[valid])["fused"].float()
        return output.reshape(B, M, embedding_dim)

    def _span_vectors(self, contextual: torch.Tensor,
                      starts: torch.Tensor, ends: torch.Tensor,
                      valid_count: int, max_span_segments: int,
                      max_span_seconds: float) -> tuple[list[CandidateSpan], torch.Tensor]:
        if valid_count < 1:
            raise ValueError("Empty candidate window")
        starts_np = starts[:valid_count].detach().cpu().numpy()
        ends_np = ends[:valid_count].detach().cpu().numpy()
        spans = enumerate_candidate_spans(
            starts_np, ends_np, max_span_segments=max_span_segments,
            max_span_seconds=max_span_seconds,
        )
        if not spans:
            spans = [CandidateSpan(
                0, 1, float(starts_np[0]), float(ends_np[0]),
            )]
        indices_start = torch.tensor([s.start_index for s in spans], device=contextual.device)
        indices_end = torch.tensor([s.end_index for s in spans], device=contextual.device)
        prefix = torch.cat([contextual.new_zeros((1, contextual.shape[-1])),
                            contextual[:valid_count].cumsum(dim=0)], dim=0)
        means = (prefix[indices_end] - prefix[indices_start]) / (
            indices_end - indices_start
        ).to(contextual.dtype).unsqueeze(-1)
        vectors = F.normalize(self.span_projection(means), dim=-1)
        return spans, vectors

    def forward(self, batch: dict, *, max_span_segments: int,
                max_span_seconds: float, align_temperature: float,
                temporal_prior_weight: float, temporal_sigma_seconds: float,
                duration_prior_weight: float) -> dict:
        segment = self.encode_segments(batch["pose"], batch["rgb"], batch["candidate_mask"])
        contextual = self.context(
            segment, batch["segment_starts"], batch["segment_ends"],
            batch["candidate_mask"],
        )
        text_vec = self.text(batch["text_embedding"])
        samples = []
        for i in range(segment.shape[0]):
            valid_count = int(batch["candidate_mask"][i].sum().item())
            spans, vectors = self._span_vectors(
                contextual[i], batch["segment_starts"][i], batch["segment_ends"][i],
                valid_count, max_span_segments, max_span_seconds,
            )
            
            semantic = vectors @ text_vec[i] / max(align_temperature, 1e-4)
            rough_start, rough_end = batch["rough_start"][i], batch["rough_end"][i]
            rough_mid = (rough_start + rough_end) / 2.0
            rough_duration = (rough_end - rough_start).clamp_min(1e-3)
            mids = torch.tensor([s.midpoint for s in spans], device=semantic.device,
                                dtype=semantic.dtype)
            durations = torch.tensor([s.duration for s in spans], device=semantic.device,
                                     dtype=semantic.dtype).clamp_min(1e-3)
            timing_prior = -0.5 * ((mids - rough_mid) / max(temporal_sigma_seconds, 1e-3)) ** 2
            timing_prior = timing_prior - duration_prior_weight * torch.abs(
                torch.log(durations / rough_duration)
            )
            final_scores = semantic + temporal_prior_weight * timing_prior
            samples.append(SampleSpans(spans, vectors, semantic, final_scores))
        return {
            "text_vectors": text_vec,
            "segment_vectors": segment,
            "contextual_vectors": contextual,
            "samples": samples,
        }
