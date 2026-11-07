from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PartialOTAlignmentLoss(nn.Module):
 

    def __init__(
        self,
        epsilon: float = 0.07,
        iterations: int = 30,
        dustbin_cost: float = 0.35,
        temporal_pair_threshold: float = 1e-6,
    ):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("Sinkhorn epsilon must be positive")
        if iterations < 1:
            raise ValueError("Sinkhorn iterations must be >= 1")
        if dustbin_cost <= 0:
            raise ValueError("Dustbin cost must be positive")
        if temporal_pair_threshold < 0:
            raise ValueError("Temporal pair threshold must be nonnegative")
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.dustbin_cost = float(dustbin_cost)
        self.temporal_pair_threshold = float(temporal_pair_threshold)

    def _sinkhorn(
        self,
        cost: torch.Tensor,
        row_mass: torch.Tensor,
        col_mass: torch.Tensor,
    ) -> torch.Tensor:
        log_kernel = -cost / self.epsilon
        log_rows = torch.log(row_mass.clamp_min(1e-12))
        log_cols = torch.log(col_mass.clamp_min(1e-12))
        log_u = torch.zeros_like(log_rows)
        log_v = torch.zeros_like(log_cols)

        for _ in range(self.iterations):
            log_u = log_rows - torch.logsumexp(
                log_kernel + log_v.unsqueeze(0), dim=1
            )
            log_v = log_cols - torch.logsumexp(
                log_kernel + log_u.unsqueeze(1), dim=0
            )

        plan = torch.exp(
            log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)
        )
        return plan

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        visual_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if visual_tokens.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError("Visual and text tokens must have shape [B,L,D]")
        if visual_tokens.shape[0] != text_tokens.shape[0]:
            raise ValueError("Visual/text batch sizes differ")
        if visual_tokens.shape[-1] != text_tokens.shape[-1]:
            raise ValueError("Visual/text embedding dimensions differ")
        if visual_mask.shape != visual_tokens.shape[:2]:
            raise ValueError("visual_mask must have shape [B,M]")
        if text_mask.shape != text_tokens.shape[:2]:
            raise ValueError("text_mask must have shape [B,T]")

        visual_mask = visual_mask.bool()
        text_mask = text_mask.bool()
        if not visual_mask.any(dim=1).all():
            raise ValueError("Every example must contain at least one visual token")

        zero = visual_tokens.float().sum() * 0.0 + text_tokens.float().sum() * 0.0
        transport_values = []
        temporal_values = []
        null_values = []
        matched_values = []

        for batch_index in range(visual_tokens.shape[0]):
            visual_indices = torch.nonzero(
                visual_mask[batch_index], as_tuple=False
            ).flatten()
            text_indices = torch.nonzero(
                text_mask[batch_index], as_tuple=False
            ).flatten()


            if text_indices.numel() == 0:
                continue

            visual = F.normalize(
                visual_tokens[batch_index, visual_indices].float(), dim=-1
            )
            text = F.normalize(
                text_tokens[batch_index, text_indices].float(), dim=-1
            )
            similarity = visual @ text.transpose(0, 1)
            real_cost = (1.0 - similarity).clamp(min=0.0, max=2.0)

            visual_count, text_count = real_cost.shape
            augmented_cost = real_cost.new_full(
                (visual_count + 1, text_count + 1), self.dustbin_cost
            )
            augmented_cost[:visual_count, :text_count] = real_cost
            augmented_cost[visual_count, text_count] = 0.0


            row_mass = torch.cat(
                (
                    real_cost.new_full((visual_count,), 1.0 / visual_count),
                    real_cost.new_ones((1,)),
                )
            )
            col_mass = torch.cat(
                (
                    real_cost.new_full((text_count,), 1.0 / text_count),
                    real_cost.new_ones((1,)),
                )
            )
            plan = self._sinkhorn(augmented_cost, row_mass, col_mass)

            total_mass = plan.sum().clamp_min(1e-8)
            transport_cost = (plan * augmented_cost).sum() / total_mass

            real_plan = plan[:visual_count, :text_count]
            row_real_mass = real_plan.sum(dim=1, keepdim=True)
            distributions = real_plan / row_real_mass.clamp_min(1e-8)

            if visual_count > 1:
                pair_is_valid = (
                    (row_real_mass[1:, 0] > self.temporal_pair_threshold)
                    & (row_real_mass[:-1, 0] > self.temporal_pair_threshold)
                )
                if pair_is_valid.any():
                    temporal = (
                        distributions[1:] - distributions[:-1]
                    ).abs().sum(dim=-1)[pair_is_valid].mean()
                else:
                    temporal = real_plan.sum() * 0.0
            else:
                temporal = real_plan.sum() * 0.0

            null_mass = (
                plan[:visual_count, text_count].sum()
                + plan[visual_count, :text_count].sum()
            )
            null_ratio = (null_mass / 2.0).clamp(0.0, 1.0)
            matched_ratio = real_plan.sum().clamp(0.0, 1.0)

            transport_values.append(transport_cost)
            temporal_values.append(temporal)
            null_values.append(null_ratio)
            matched_values.append(matched_ratio)

        if not transport_values:
            return {
                "loss": zero,
                "transport_cost": zero,
                "temporal_variation": zero,
                "null_ratio": zero,
                "matched_ratio": zero,
            }

        transport_mean = torch.stack(transport_values).mean()
        temporal_mean = torch.stack(temporal_values).mean()
        null_mean = torch.stack(null_values).mean()
        matched_mean = torch.stack(matched_values).mean()
        return {
            "loss": transport_mean + temporal_mean,
            "transport_cost": transport_mean,
            "temporal_variation": temporal_mean,
            "null_ratio": null_mean,
            "matched_ratio": matched_mean,
        }
