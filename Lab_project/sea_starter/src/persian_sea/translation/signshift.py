

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


STREAM_NAMES = ("signer", "hands", "face")


class MultiScaleTemporalDifference(nn.Module):


    def __init__(self, dim: int, offsets: tuple[int, ...], dropout: float = 0.1):
        super().__init__()
        if dim < 1 or not offsets or any(x < 1 for x in offsets):
            raise ValueError("Invalid TDM dimension or offsets")
        self.offsets = tuple(sorted(set(int(x) for x in offsets)))
        self.delta_projection = nn.Conv1d(dim, dim, kernel_size=1)
        self.gate = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2 or seq.shape[0] < 1:
            raise ValueError("TDM expects a nonempty [segments, dim] tensor")
        
        differences = []
        for d in self.offsets:
            if d >= seq.size(0):
                continue
            delta = torch.zeros_like(seq)
            delta[d:] = seq[d:] - seq[:-d]
            differences.append(delta)
        if not differences:
            return self.norm(seq)
        mean_delta = torch.stack(differences, dim=0).mean(dim=0)
        delta_ch = mean_delta.transpose(0, 1).unsqueeze(0)
        projected = self.delta_projection(delta_ch)
        gate = torch.sigmoid(self.gate(delta_ch))
        enhanced = (gate * projected).squeeze(0).transpose(0, 1)
        return self.norm(seq + self.dropout(enhanced))


class LocalStreamFusion(nn.Module):


    def __init__(
        self, local_dim: int, hidden_dim: int,
        offsets: tuple[int, ...], dropout: float,
    ):
        super().__init__()
        if local_dim < 1:
            raise ValueError("local_dim must be positive when using local streams")
        self.local_dim = int(local_dim)
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(local_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            for name in STREAM_NAMES
        })
        self.tdm = nn.ModuleDict({
            name: MultiScaleTemporalDifference(hidden_dim, offsets, dropout)
            for name in STREAM_NAMES
        })
        self.raw_gates = nn.Parameter(torch.full((len(STREAM_NAMES),), -1.0))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, global_hidden: torch.Tensor, streams: dict) -> torch.Tensor:
        if not isinstance(streams, dict):
            raise ValueError("Multi-stream mode needs signer/hands/face features")
        gates = torch.sigmoid(self.raw_gates)
        fused = global_hidden
        for index, name in enumerate(STREAM_NAMES):
            if name not in streams:
                raise ValueError(f"Missing required local feature stream: {name}")
            values = streams[name]
            if values.shape != (global_hidden.shape[0], self.local_dim):
                raise ValueError(
                    f"Bad {name} shape: {tuple(values.shape)}; expected "
                    f"({len(global_hidden)}, {self.local_dim})"
                )
            if not torch.isfinite(values).all():
                raise ValueError(f"Nonfinite local features: {name}")
            projected = self.projections[name](values.float())
            fused = fused + gates[index] * self.tdm[name](projected)
        return self.norm(fused / (1.0 + gates.sum()))


class SegmentCountHead(nn.Module):


    def __init__(
        self, hidden_dim: int, max_sentences: int,
        conv_layers: int = 4, dropout: float = 0.1,
    ):
        super().__init__()
        if not (max_sentences >= 1 and conv_layers >= 1):
            raise ValueError("Invalid SCP limits")
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2**layer,
                          dilation=2**layer),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for layer in range(conv_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_sentences),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or len(hidden) < 1:
            raise ValueError("SCP expects [segments, hidden_dim]")
        x = hidden.transpose(0, 1).unsqueeze(0)
        for block in self.blocks:
            x = x + block(x)
        x = self.norm(x.squeeze(0).transpose(0, 1))
        weights = torch.softmax(self.attention(x).squeeze(-1), dim=0)
        pooled = (x * weights[:, None]).sum(dim=0)

        return self.classifier(pooled)


def count_kl_loss(logits: torch.Tensor, count: int, sigma: float = 1.0) -> torch.Tensor:
    if logits.ndim != 1 or not 1 <= count <= logits.numel():
        raise ValueError(
            f"Count target {count} exceeds available SCP classes (1..{logits.numel()})"
        )
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError("SCP sigma must be finite and positive")
    positions = torch.arange(1, logits.numel() + 1, device=logits.device,
                             dtype=torch.float32)
    soft_target = torch.softmax(-((positions - count) ** 2) / (2 * sigma**2), dim=0)
    return F.kl_div(F.log_softmax(logits.float(), dim=0),
                    soft_target, reduction="sum")


def temporal_smoothness_loss(boundary_logits: torch.Tensor) -> torch.Tensor:
    if boundary_logits.ndim != 2 or boundary_logits.shape[1] != 2:
        raise ValueError("Smoothness expects [segments, 2] logits")
    if len(boundary_logits) < 2:
        return boundary_logits.sum() * 0.0
    probs = torch.sigmoid(boundary_logits)
    return (probs[1:] - probs[:-1]).square().mean()


@dataclass(frozen=True)
class CountDecodeInfo:
    predicted_count: int
    selected_count: int
    predicted_count_probability: float
    constrained: bool
    adjustment_reason: str


def decode_count_constrained(
    starts: np.ndarray,
    ends: np.ndarray,
    start_probs: np.ndarray,
    end_probs: np.ndarray,
    count_logits: np.ndarray,
    *,
    min_segments: int,
    max_segments: int,
) -> tuple[list[dict], CountDecodeInfo]:


    starts, ends = np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)
    ps = np.asarray(start_probs, dtype=np.float64).reshape(-1)
    pe = np.asarray(end_probs, dtype=np.float64).reshape(-1)
    counts = np.asarray(count_logits, dtype=np.float64).reshape(-1)
    n = len(starts)
    if not (n > 0 and len(ends) == n and len(ps) == n and len(pe) == n
            and counts.size >= 1 and 1 <= min_segments <= max_segments):
        raise ValueError("Invalid count-decoding inputs")
    if not all(np.isfinite(x).all() for x in (starts, ends, ps, pe, counts)):
        raise ValueError("Count decoder received nonfinite inputs")
    if (np.any(ends <= starts) or np.any(np.diff(starts) < 0)
            or np.any((ps < 0) | (ps > 1)) or np.any((pe < 0) | (pe > 1))):
        raise ValueError("Bad segment order or probabilities")



    if n < min_segments:
        normalized = np.exp(counts - counts.max())
        normalized /= normalized.sum()
        requested = int(np.argmax(counts)) + 1
        return ([{
            "start_index": 0,
            "end_index": n - 1,
            "start": float(starts[0]),
            "end": float(ends[-1]),
            "start_confidence": float(ps[0]),
            "end_confidence": float(pe[-1]),
            "num_sign_segments": n,
        }], CountDecodeInfo(
            predicted_count=requested,
            selected_count=1,
            predicted_count_probability=float(normalized[requested - 1]),
            constrained=True,
            adjustment_reason=(
                f"Video has {n} sign segment(s), shorter than "
                f"min_segments={min_segments}; returning a single window"
            ),
        ))

    feasible_min = max(1, int(math.ceil(n / max_segments)))
    feasible_max = max(1, int(n // min_segments))
    if feasible_min > feasible_max:
        raise ValueError(
            f"No full-cover sentence partition for N={n}, "
            f"min_segments={min_segments}, max_segments={max_segments}; "
            "review boundary length constraints"
        )
    normalized = np.exp(counts - counts.max())
    normalized /= normalized.sum()
    requested = int(np.argmax(counts)) + 1
    k = int(np.clip(requested, feasible_min, feasible_max))
    adjusted = (k != requested)


    eps = 1e-5
    left = np.log(np.clip(pe, eps, 1 - eps) / (1 - np.clip(pe, eps, 1 - eps)))
    right = np.log(np.clip(ps, eps, 1 - eps) / (1 - np.clip(ps, eps, 1 - eps)))
    edge = np.zeros(n + 1, dtype=np.float64)
    edge[1:n] = 0.5 * (left[:-1] + right[1:])

    dp = np.full((k + 1, n + 1), -np.inf, dtype=np.float64)
    back = np.full((k + 1, n + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for used in range(1, k + 1):
        low = max(used * min_segments, n - (k - used) * max_segments)
        high = min(used * max_segments, n - (k - used) * min_segments)
        for stop in range(low, high + 1):
            first = max((used - 1) * min_segments, stop - max_segments)
            last = min((used - 1) * max_segments, stop - min_segments)
            if first > last:
                continue
            choices = dp[used - 1, first:last + 1]
            if choices.size == 0 or not np.isfinite(choices).any():
                continue
            winner = first + int(np.argmax(choices))
            dp[used, stop] = dp[used - 1, winner] + (edge[stop] if used < k else 0.0)
            back[used, stop] = winner
    if not np.isfinite(dp[k, n]):
        raise RuntimeError("Count DP could not reconstruct a full partition")

    slices = []
    stop = n
    for used in range(k, 0, -1):
        begin = int(back[used, stop])
        if begin < 0:
            raise RuntimeError("Broken count decoder backpointer")
        slices.append((begin, stop))
        stop = begin
    slices.reverse()
    if stop != 0 or slices[0][0] != 0 or slices[-1][1] != n:
        raise RuntimeError("Count decoder left unassigned sign segments")

    windows = [{
        "start_index": begin,
        "end_index": stop - 1,
        "start": float(starts[begin]),
        "end": float(ends[stop - 1]),
        "start_confidence": float(ps[begin]),
        "end_confidence": float(pe[stop - 1]),
        "num_sign_segments": stop - begin,
    } for begin, stop in slices]
    return windows, CountDecodeInfo(
        predicted_count=requested,
        selected_count=k,
        predicted_count_probability=float(normalized[requested - 1]),
        constrained=adjusted,
        adjustment_reason=(
            f"Predicted {requested} not feasible for {n} sign segments with "
            f"length [{min_segments}, {max_segments}]; used {k}"
            if adjusted else ""
        ),
    )


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / union if union > 0 else 0.0


def sentence_segmentation_metrics(
    predicted: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    thresholds: tuple[float, ...] = (0.10, 0.25, 0.50),
) -> dict:


    if not reference:
        raise ValueError("Cannot evaluate without reference sentence boundaries")
    for pair in (*predicted, *reference):
        if len(pair) != 2 or not np.isfinite(pair).all() or pair[1] <= pair[0]:
            raise ValueError("Invalid predicted/reference sentence interval")
    scores = {}
    for t in thresholds:
        candidates = sorted(
            [(_interval_iou(a, b), i, j)
             for i, a in enumerate(predicted)
             for j, b in enumerate(reference)
             if _interval_iou(a, b) >= t],
            reverse=True,
        )
        used_pred, used_ref = set(), set()
        for _, i, j in candidates:
            if i not in used_pred and j not in used_ref:
                used_pred.add(i)
                used_ref.add(j)
        tp = len(used_pred)
        fp = len(predicted) - tp
        fn = len(reference) - tp
        scores[f"f1_{int(round(t * 100))}"] = 2 * tp / max(1, 2 * tp + fp + fn)
        scores[f"tp_{int(round(t * 100))}"] = tp
        scores[f"fp_{int(round(t * 100))}"] = fp
        scores[f"fn_{int(round(t * 100))}"] = fn
    count_error = abs(len(predicted) - len(reference))
    scores.update({
        "ser": count_error / max(1, len(reference)),
        "count_mae": float(count_error),
        "predicted_sentences": len(predicted),
        "reference_sentences": len(reference),
    })
    return scores


def aggregate_sentence_metrics(items: list[dict]) -> dict:
    if not items:
        raise ValueError("No videos were evaluated")
    result = {"videos": len(items)}
    for t in (10, 25, 50):
        tp = sum(row[f"tp_{t}"] for row in items)
        fp = sum(row[f"fp_{t}"] for row in items)
        fn = sum(row[f"fn_{t}"] for row in items)
        result[f"f1_{t}"] = 2 * tp / max(1, 2 * tp + fp + fn)
        result[f"tp_{t}"] = tp
        result[f"fp_{t}"] = fp
        result[f"fn_{t}"] = fn
    result["ser"] = float(np.mean([row["ser"] for row in items]))
    result["count_mae"] = float(np.mean([row["count_mae"] for row in items]))
    result["count_bias"] = float(np.mean([
        row["predicted_sentences"] - row["reference_sentences"] for row in items
    ]))
    return result
