from types import SimpleNamespace
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

from persian_sea.translation.model import PersianTranslator


class DummyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 8)

    def forward(self, pose, rgb, starts, ends, mask):
        del rgb, starts, ends
        pooled = pose.mean(dim=(2, 3))
        return self.proj(pooled) * mask[..., None]


class DummyTextEncoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids, attention_mask, return_dict=True):
        del return_dict
        hidden = self.embedding(input_ids)
        hidden = hidden * attention_mask[..., None]
        return SimpleNamespace(last_hidden_state=hidden)


class DummySeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=12, pad_token_id=0)
        self.embedding = nn.Embedding(64, 12)
        self.encoder = DummyTextEncoder(self.embedding)
        self.lm_head = nn.Linear(12, 64)

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        encoder_outputs=None,
        attention_mask=None,
        labels=None,
        input_ids=None,
        use_cache=None,
    ):
        del use_cache
        if input_ids is not None:
            source = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        else:
            source = encoder_outputs.last_hidden_state

        pooled = source.mean(dim=1)
        steps = labels.shape[1]
        logits = self.lm_head(pooled).unsqueeze(1).expand(-1, steps, -1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, encoder_outputs, attention_mask, max_new_tokens):
        del attention_mask, max_new_tokens
        return torch.ones(
            (encoder_outputs.last_hidden_state.shape[0], 2),
            dtype=torch.long,
        )


def _install_transformers_output_stub(monkeypatch):
    if "transformers" in sys.modules:
        return
    package = types.ModuleType("transformers")
    package.__path__ = []
    results = types.ModuleType("transformers.modeling_outputs")
    results.BaseModelOutput = lambda last_hidden_state: SimpleNamespace(
        last_hidden_state=last_hidden_state
    )
    monkeypatch.setitem(sys.modules, "transformers", package)
    monkeypatch.setitem(sys.modules, "transformers.modeling_outputs", results)


def test_model_combines_ce_partial_ot_temporal_and_tpa(monkeypatch):
    _install_transformers_output_stub(monkeypatch)
    student = DummySeq2Seq()
    teacher = DummySeq2Seq()
    model = PersianTranslator(
        DummyVisual(),
        8,
        student,
        freeze_visual=False,
        ota_enabled=True,
        ota_weight=0.1,
        ota_epsilon=0.1,
        ota_iterations=20,
        ota_dustbin_cost=0.6,
        ota_temporal_weight=0.01,
        prior_teacher=teacher,
        tpa_weight=0.02,
        tpa_temperature=1.0,
    )

    batch = {
        "pose": torch.randn(2, 4, 2, 75, 4),
        "rgb": torch.randn(2, 4, 2, 1, 8),
        "starts": torch.tensor([[0., 1., 2., 3.], [0., 1., 2., 0.]]),
        "ends": torch.tensor([[.8, 1.8, 2.8, 3.8], [.8, 1.8, 2.8, 0.]]),
        "mask": torch.tensor([
            [True, True, True, True],
            [True, True, True, False],
        ]),
        "labels": torch.tensor([[4, 5, 6, 1], [7, 8, 1, -100]]),
        "target_input_ids": torch.tensor([[4, 5, 6, 1], [7, 8, 1, 0]]),
        "target_attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "target_content_mask": torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
        ]),
    }

    model.train()
    output = model(batch)
    for name in (
        "loss",
        "ce_loss",
        "ota_loss",
        "temporal_variation_loss",
        "tpa_loss",
        "null_ratio",
        "matched_ratio",
    ):
        assert torch.isfinite(getattr(output, name))

    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.adapter.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert teacher.training is False
