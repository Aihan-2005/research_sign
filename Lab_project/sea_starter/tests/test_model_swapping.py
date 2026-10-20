

import torch
from torch import nn
import torch.nn.functional as F

from persian_sea.model.builders import visual_backends
from persian_sea.semantic.model import SemanticAligner, CandidateContextEncoder


class TinyVisual(nn.Module):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__()
        self.projection = nn.Linear(4, output_dim)

    def forward(self, pose, rgb):
        pooled = pose.mean(dim=(1, 2))
        return {"fused": F.normalize(self.projection(pooled), dim=-1)}


def test_injected_visual_backend_learns_with_same_alignment_interface():
    visual_backends.register('tiny_test_only', lambda **kwargs: TinyVisual(**kwargs))
    model = SemanticAligner(text_input_dim=16, rgb_input_dim=4, num_views=1,
                            embedding_dim=8, visual_backend='tiny_test_only')
    model.train()
    batch = {
        'pose': torch.randn(2, 3, 2, 75, 4),
        'rgb': torch.randn(2, 3, 2, 1, 4),
        'candidate_mask': torch.tensor([[True, True, False], [True, True, True]]),
        'segment_starts': torch.tensor([[1., 2., 0.], [8., 9., 10.]]),
        'segment_ends': torch.tensor([[1.5, 2.5, 0.], [8.5, 9.5, 10.5]]),
        'text_embedding': torch.randn(2, 16),
        'rough_start': torch.tensor([1., 8.]),
        'rough_end': torch.tensor([2.5, 10.5]),
    }
    result = model(batch, max_span_segments=3, max_span_seconds=3.,
                   align_temperature=0.07, temporal_prior_weight=0.3,
                   temporal_sigma_seconds=2., duration_prior_weight=0.2)
    assert len(result['samples']) == 2
    assert result['samples'][0].vectors.shape[1] == 8
    result['samples'][0].final_scores.sum().backward()
    assert model.visual.projection.weight.grad is not None


def test_context_embeddings_independent_of_other_padding_length():
    model = CandidateContextEncoder(dim=8, layers=1, heads=2, dropout=0.)
    model.eval()
    data = torch.randn(1, 2, 8)
    st = torch.tensor([[10., 11.]])
    en = torch.tensor([[10.5, 11.5]])
    a = model(data, st, en, torch.ones(1, 2, dtype=torch.bool))
    padded = torch.cat([data, torch.zeros(1, 3, 8)], dim=1)
    st2 = torch.cat([st, torch.zeros(1, 3)], dim=1)
    en2 = torch.cat([en, torch.zeros(1, 3)], dim=1)
    b = model(padded, st2, en2, torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool))
    assert torch.allclose(a, b[:, :2], atol=1e-5)
