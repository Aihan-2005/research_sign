from __future__ import annotations

import torch
from torch import nn


def _adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    A = torch.eye(num_nodes, dtype=torch.float32)
    for i, j in edges:
        if i < num_nodes and j < num_nodes:
            A[i, j] = 1.0
            A[j, i] = 1.0
    degree = A.sum(dim=1).clamp_min(1.0)
    d_inv_sqrt = degree.pow(-0.5)
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


# MediaPipe hand graph: wrist=0; four joints per finger.
HAND_EDGES = []
for chain in ([0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]):
    HAND_EDGES += list(zip(chain[:-1], chain[1:]))

# Useful upper-body/arm subset of MediaPipe Pose graph; all 33 nodes remain.
BODY_EDGES = [
    (11,12), (11,13), (13,15), (12,14), (14,16),
    (15,17), (15,19), (15,21), (16,18), (16,20), (16,22),
    (11,23), (12,24), (23,24), (23,25), (24,26),
    (25,27), (26,28), (27,29), (28,30), (29,31), (30,32),
]


class GraphBlock(nn.Module):
    def __init__(self, dim: int, adjacency: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("A", adjacency)
        self.pre = nn.Linear(dim, dim)
        self.post = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,D]
        h = self.pre(x)
        h = torch.einsum("vw,btwd->btvd", self.A, h)
        return x + self.post(h)


class PartGraphEncoder(nn.Module):
    def __init__(self, num_nodes: int, edges: list[tuple[int,int]], in_dim: int, dim: int):
        super().__init__()
        A = _adjacency(num_nodes, edges)
        self.input = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([
            GraphBlock(dim, A),
            GraphBlock(dim, A),
        ])
        self.node_attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        score = self.node_attention(h).squeeze(-1)
        score = torch.softmax(score, dim=-1)
        return torch.einsum("btv,btvd->btd", score, h)


class PoseSemanticEncoder(nn.Module):
    """Trainable encoder for cached [B,K,75,C] normalized MediaPipe pose."""
    def __init__(
        self,
        input_dim: int = 4,
        part_dim: int = 128,
        model_dim: int = 384,
        temporal_layers: int = 3,
        temporal_heads: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.body = PartGraphEncoder(33, BODY_EDGES, input_dim, part_dim)
        self.left = PartGraphEncoder(21, HAND_EDGES, input_dim, part_dim)
        # Weight sharing is intentional: left/right hands have the same topology.
        self.right = self.left

        self.part_fusion = nn.Sequential(
            nn.Linear(part_dim * 3, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=temporal_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.pool_query = nn.Parameter(torch.randn(model_dim) * 0.02)

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pose [B,K,75,C]
        body = pose[:, :, :33]
        left = pose[:, :, 33:54]
        right = pose[:, :, 54:75]

        body_h = self.body(body)
        left_h = self.left(left)
        right_h = self.right(right)

        h = self.part_fusion(torch.cat([body_h, left_h, right_h], dim=-1))
        h = self.temporal(h)

        score = torch.einsum("btd,d->bt", h, self.pool_query)
        score = torch.softmax(score, dim=-1)
        pooled = torch.einsum("bt,btd->bd", score, h)
        return pooled, h
