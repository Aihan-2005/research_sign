from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from persian_sea.semantic.candidates import interval_iou


def mil_bag_score(
    text_vector: torch.Tensor,
    span_vectors: torch.Tensor,
    mil_temperature: float,
) -> torch.Tensor:


    if span_vectors.ndim != 2 or span_vectors.size(0) == 0:
        raise ValueError("span_vectors must be a nonempty [N,D] tensor")
    tau = max(float(mil_temperature), 1e-4)
    sims = span_vectors @ text_vector
    return tau * (torch.logsumexp(sims / tau, dim=0) - math.log(sims.numel()))


def contrastive_bag_loss(
    text_vectors: torch.Tensor,
    sample_outputs: list,
    video_ids: list[str],
    mil_temperature: float,
    contrastive_temperature: float,
    rough_start: torch.Tensor | None = None,
    rough_end: torch.Tensor | None = None,
    same_video_min_distance_seconds: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:


    B = text_vectors.shape[0]
    if B != len(sample_outputs) or B != len(video_ids):
        raise ValueError("Batch metadata length mismatch")
    if B < 2:
        zero = text_vectors.sum() * 0.0
        return zero, text_vectors.new_zeros((B, B))

    matrix = torch.stack([
        torch.stack([
            mil_bag_score(text_vectors[i], sample_outputs[j].vectors, mil_temperature)
            for j in range(B)
        ])
        for i in range(B)
    ])

    allowed = torch.ones((B, B), dtype=torch.bool, device=matrix.device)
    for i in range(B):
        for j in range(B):
            if i == j or video_ids[i] != video_ids[j]:
                continue
            if rough_start is None or rough_end is None:
                allowed[i, j] = False
                continue

            gap = max(
                0.0,
                float((rough_start[j] - rough_end[i]).item()),
                float((rough_start[i] - rough_end[j]).item()),
            )
            allowed[i, j] = gap >= same_video_min_distance_seconds

    logits = matrix / max(float(contrastive_temperature), 1e-4)
    logits = logits.masked_fill(~allowed, -1e4)
    target = torch.arange(B, device=logits.device)


    informative = allowed.sum(dim=1) > 1
    informative_columns = allowed.sum(dim=0) > 1
    terms = []
    if informative.any():
        terms.append(F.cross_entropy(logits[informative], target[informative]))
    if informative_columns.any():
        terms.append(F.cross_entropy(logits.T[informative_columns], target[informative_columns]))
    loss = torch.stack(terms).mean() if terms else matrix.sum() * 0.0
    return loss, matrix.detach()


def supervised_span_loss(
    sample_outputs: list,
    gt_start: torch.Tensor,
    gt_end: torch.Tensor,
    iou_temperature: float,
) -> tuple[torch.Tensor, int]:


    losses = []
    tau = max(float(iou_temperature), 1e-4)
    for i, output in enumerate(sample_outputs):
        if not (bool(torch.isfinite(gt_start[i])) and bool(torch.isfinite(gt_end[i]))):
            continue
        gs, ge = float(gt_start[i].item()), float(gt_end[i].item())
        if not (0 <= gs < ge):
            continue
        ious = torch.tensor(
            [interval_iou(s.start_time, s.end_time, gs, ge) for s in output.spans],
            dtype=output.final_scores.dtype,
            device=output.final_scores.device,
        )
        positive = ious > 0
        if not positive.any():

            continue
        target = torch.zeros_like(ious)
        target[positive] = torch.softmax(ious[positive] / tau, dim=0)
        log_prob = F.log_softmax(output.final_scores, dim=0)
        losses.append(-(target * log_prob).sum())

    if not losses:

        zero = sum((o.final_scores.sum() * 0.0 for o in sample_outputs), gt_start.new_tensor(0.0))
        return zero, 0
    return torch.stack(losses).mean(), len(losses)
