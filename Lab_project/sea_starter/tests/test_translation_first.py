

from __future__ import annotations

from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from torch import nn

from persian_sea.io.subtitles import SubtitleCue, write_vtt
from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.data import (TranslationCueDataset, collate,
                                          inference_windows, validate_cache)
from persian_sea.translation.model import PersianTranslator, VisualLanguageAdapter


class DummyTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        del padding, truncation, max_length, return_tensors
        ids = torch.full((len(texts), 3), 7, dtype=torch.long)
        mask = torch.ones_like(ids)
        if len(texts) > 1:
            ids[1, 2] = 0
            mask[1, 2] = 0
        return {"input_ids": ids, "attention_mask": mask}


class DummyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 8)

    def forward(self, pose, rgb, starts, ends, mask):
        del rgb, starts, ends

        mean = pose.mean(dim=(2, 3))
        return self.proj(mean) * mask[..., None]


class DummySeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=12)
        self.out = nn.Linear(12, 1)

    def forward(self, encoder_outputs, attention_mask, labels):
        memory = encoder_outputs.last_hidden_state
        assert memory.shape[:2] == attention_mask.shape
        assert (memory[attention_mask == 0] == 0).all()
        score = self.out(memory).sum() / attention_mask.sum()
        return SimpleNamespace(loss=(score - labels[labels != -100].float().mean())**2)

    def generate(self, encoder_outputs, attention_mask, max_new_tokens):
        assert max_new_tokens > 0
        assert encoder_outputs.last_hidden_state.shape[:2] == attention_mask.shape
        return torch.tensor([[5, 6]], dtype=torch.long)


def fake_visual_cache(path, n=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pose=np.ones((n, 2, 75, 4), dtype=np.float16),
                        rgb=np.ones((n, 2, 1, 8), dtype=np.float16),
                        starts=np.arange(n, dtype=np.float32),
                        ends=np.arange(1, n + 1, dtype=np.float32))


def test_cfg_rejects_broken_windows():
    with pytest.raises(ValueError, match="Stride"):
        TranslationConfig(inference_stride_segments=40, max_source_segments=32).validate()
    with pytest.raises(ValueError, match="signer_metadata"):
        TranslationConfig(strict_signer=True).validate()


def test_dataset_and_inference_do_not_share_subtitle_dependency(tmp_path):
    subs = tmp_path / "subtitles"
    subs.mkdir()
    write_vtt([SubtitleCue(1., 2.5, "سلام"), SubtitleCue(4., 5.5, "حال شما چطور است")],
              subs / "a.vtt")
    cache = tmp_path / "features"
    fake_visual_cache(cache / "a.npz")
    ids = tmp_path / "ids.txt"
    ids.write_text("a\n", encoding="utf-8")
    cfg = TranslationConfig(subtitles_dir=str(subs), visual_cache_dir=str(cache),
                            max_source_segments=5, inference_stride_segments=3)
    ds = TranslationCueDataset(cfg, ids)
    assert len(ds) == 2
    a, b = ds[0], ds[1]
    assert a["text"] == "سلام"
    batch = collate([a, b], DummyTokenizer(), max_target_tokens=16)
    assert batch["mask"].shape[0] == 2
    assert batch["labels"][1, 2].item() == -100
    with np.load(cache / "a.npz", allow_pickle=False) as npz:
        data = {k: npz[k] for k in ("starts", "ends", "pose", "rgb")}
    windows = inference_windows(data, window_size=5, stride=3)
    assert list(windows[0]) == [0, 1, 2, 3, 4]
    assert list(windows[-1])[-1] == 7
    (subs / "a.vtt").unlink()
    assert len(inference_windows(data, 5, 3)) == 3


def test_cache_rejects_nan(tmp_path):
    f = tmp_path / "feature.npz"
    fake_visual_cache(f)
    with np.load(f, allow_pickle=False) as npz:
        d = {k: npz[k].copy() for k in ("pose", "rgb", "starts", "ends")}
    d["pose"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_cache(d)


def test_adapter_zeroes_padding():
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    y = VisualLanguageAdapter(8, 12)(x, mask)
    assert y.shape == (2, 4, 12)
    assert torch.count_nonzero(y[~mask]) == 0


def test_decoder_forward_generate_without_text_input(monkeypatch):
    import types
    import importlib.util
    if importlib.util.find_spec("transformers") is None:
        package = types.ModuleType("transformers")
        package.__path__ = []
        results = types.ModuleType("transformers.modeling_outputs")
        results.BaseModelOutput = lambda last_hidden_state: SimpleNamespace(
            last_hidden_state=last_hidden_state)
        monkeypatch.setitem(sys.modules, "transformers", package)
        monkeypatch.setitem(sys.modules, "transformers.modeling_outputs", results)

    model = PersianTranslator(DummyVisual(), 8, DummySeq2Seq(), freeze_visual=True)
    data = {"pose": torch.randn(1, 3, 2, 75, 4), "rgb": torch.randn(1, 3, 2, 1, 8),
            "starts": torch.tensor([[0., 1., 0.]]), "ends": torch.tensor([[1., 2., 0.]]),
            "mask": torch.tensor([[True, True, False]]),
            "labels": torch.tensor([[4, 5, 6]])}
    model.train()
    out = model(data)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.adapter.parameters())
    assert all(p.grad is None for p in model.visual.parameters())
    assert model.visual.training is False
    ids = model.generate({key: value for key, value in data.items() if key != "labels"})
    assert ids.shape == (1, 2)


def test_readiness_rejects_missing_vlp_and_accepts_synthetic_split(tmp_path):
    from persian_sea.translation.readiness import readiness_report
    subtitles, features = tmp_path / "subtitles", tmp_path / "features"
    subtitles.mkdir()
    ids = {}
    for split, videos in (("train", ["a", "b"]), ("val", ["c"]), ("test", ["d"])):
        split_path = tmp_path / (split + ".txt")
        split_path.write_text("\n".join(videos) + "\n")
        ids[split] = str(split_path)
        for vid in videos:
            write_vtt([SubtitleCue(0., 1., "سلام")], subtitles / f"{vid}.vtt")
            fake_visual_cache(features / f"{vid}.npz")
    cfg = TranslationConfig(
        train_ids=ids["train"], val_ids=ids["val"], test_ids=ids["test"],
        subtitles_dir=str(subtitles), visual_cache_dir=str(features),
        visual_checkpoint=str(tmp_path / "absent.pt"),
        require_pretrained_visual=True,
    )
    blocked = readiness_report(cfg)
    assert not blocked["ok"]
    assert any("checkpoint" in message for message in blocked["errors"])
    from dataclasses import replace
    allowed = readiness_report(replace(cfg, require_pretrained_visual=False))
    assert allowed["ok"]
    assert allowed["checked"]["train"]["videos"] == 2
    assert allowed["visual_shapes_without_video_axis"] == [[2, 75, 4], [2, 1, 8]]
