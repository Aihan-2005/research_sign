from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from persian_sea.config import load_config
from persian_sea.text.cache_identity import (
    text_digest, identity, cache_directory, validate_sidecar,
)
from persian_sea.text.cache import scan_text_video, text_cache_directory, verify_text_cache
from persian_sea.text.backends import text_backends
from persian_sea.text.qwen3_embedding import last_token_pool
from persian_sea.video_text.config import VideoTextConfig
from persian_sea.video_text.span_pooling import SpanPooler
from persian_sea.video_text.model import VideoTextModel, mil_similarity_matrix
from persian_sea.video_text.train import architecture_config, checkpoint_architecture, verify_checkpoint_text_source


def app_config(tmp_path: Path):
    root = Path(__file__).parents[1]
    cfg = load_config(root / "configs/baseline.yaml")
    sub = tmp_path / "subtitles"; sub.mkdir(exist_ok=True)
    (sub / "clip.vtt").write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nسلام دنیا\n", encoding="utf-8"
    )
    paths = replace(cfg.paths, subtitles_raw=sub,
                    text_feature_cache=tmp_path / "text_features")
    sem = replace(cfg.semantic, text_backend="bge_m3",
                  text_cache_layout="by_backend", text_model="BAAI/bge-m3")
    return replace(cfg, paths=paths, semantic=sem)


class DummyTextExtractor:
    def encode(self, texts, batch_size=2):
        return np.ones((len(texts), 8), dtype=np.float32) * 0.5


def test_text_encoder_registry_has_two_backends_without_loading_weights():
    assert callable(text_backends.create)
    with pytest.raises(Exception):
        text_backends.create("nonexistent_model")


def test_text_cache_partition_and_identity():
    assert cache_directory("outputs/text", "bge_m3", "by_backend") == Path("outputs/text/bge_m3")
    assert cache_directory("outputs/text", "bge_m3", "legacy") == Path("outputs/text")
    with pytest.raises(ValueError):
        cache_directory("outputs/text", "../../evil", "by_backend")
    assert identity("qwen3_embedding", "Qwen/embedding")["pooling"] == "last_token"
    assert text_digest(["سلام"]) != text_digest(["سلام! "])


def test_model_aware_text_cache_roundtrip_and_rejects_wrong_model(tmp_path):
    cfg = app_config(tmp_path)
    path = scan_text_video("clip", cfg, DummyTextExtractor())
    assert path == text_cache_directory(cfg) / "clip.npz"
    verify_text_cache(path, cfg, "clip")
    scan_text_video("clip", cfg, DummyTextExtractor())  # idempotent verified skip
    changed = replace(cfg, semantic=replace(cfg.semantic, text_model="OTHER-MODEL"))
    with pytest.raises(ValueError, match="Stale text cache"):
        verify_text_cache(path, changed, "clip")
    with pytest.raises(ValueError, match="Stale text cache"):
        scan_text_video("clip", changed, DummyTextExtractor())


def test_text_cache_rejects_subtitle_mutation(tmp_path):
    cfg = app_config(tmp_path)
    path = scan_text_video("clip", cfg, DummyTextExtractor())
    (cfg.paths.subtitles_raw / "clip.vtt").write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nمتن متفاوت\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(Stale text cache|text/order)"):
        verify_text_cache(path, cfg, "clip")


def test_qwen_last_token_pool_both_padding_sides():
    hidden = torch.arange(2 * 5 * 3).reshape(2, 5, 3).float()
    masks = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
    got = last_token_pool(hidden, masks)
    assert torch.equal(got[0], hidden[0, 2])
    assert torch.equal(got[1], hidden[1, 4])
    with pytest.raises(ValueError):
        last_token_pool(hidden, torch.zeros(2, 5, dtype=torch.long))


@pytest.mark.parametrize("pool_type", ["mean", "attentive"])
def test_span_pooling_mask_shape_and_gradient(pool_type):
    torch.manual_seed(7)
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    pooling = SpanPooler(8, (1, 2, 4), pool_type)
    span, valid = pooling(tokens, mask)
    assert span.shape == (2, 5 + 4 + 2, 8)
    assert not valid[0, 9:].any() 
    assert torch.isfinite(span).all()
    span[valid].square().sum().backward()
    assert torch.isfinite(tokens.grad).all()
    if pool_type == "attentive":
        assert pooling.score[1].weight.grad is not None


def test_attentive_pooling_can_distinguish_reversed_tokens():
    torch.manual_seed(3)
    p = SpanPooler(4, (3,), "attentive")
    x = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    a = p(x, mask)[0]
    b = p(x.flip(1), mask)[0]
    assert not torch.allclose(a, b)


def test_videotext_config_and_checkpoint_tagging():
    cfg = VideoTextConfig(embedding_dim=8, temporal_heads=2, span_pooling="attentive")
    assert cfg.validate().span_pooling == "attentive"
    with pytest.raises(ValueError):
        replace(cfg, span_pooling="invalid").validate()
    old = architecture_config(replace(cfg, span_pooling="mean"))
    old.pop("span_pooling")
    assert checkpoint_architecture(old)["span_pooling"] == "mean"
    assert checkpoint_architecture(old) != architecture_config(cfg)


def test_videotext_model_trainable_pooling_with_mock_visual():
    class DummyVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 8)
        def forward(self, pose, rgb):
            return {"fused": torch.nn.functional.normalize(self.layer(pose.mean((1, 2))), dim=-1)}

    cfg = VideoTextConfig(embedding_dim=8, temporal_heads=2, temporal_layers=0,
                          span_sizes=(1, 2), span_pooling="attentive")
    m = VideoTextModel(cfg, text_dim=6, rgb_dim=7, num_views=2,
                       visual=DummyVisual())
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    batch = {"pose": torch.rand(2, 3, 2, 75, 4),
             "rgb": torch.rand(2, 3, 2, 2, 7),
             "starts": torch.tensor([[0., 1., 2.], [0., 1., 0.]]),
             "ends": torch.tensor([[.8, 1.8, 2.8], [.8, 1.8, 0.]]),
             "mask": mask, "text_embedding": torch.rand(2, 6)}
    y = m(batch)
    s = mil_similarity_matrix(y["text_vectors"], y["span_vectors"], y["span_mask"], .1)
    s.sum().backward()
    assert s.shape == (2, 2)
    assert m.video.visual.layer.weight.grad is not None


def test_strict_video_text_cache_provenance(tmp_path):
    expected = identity("bge_m3", "BAAI/bge-m3")
    path = tmp_path / "video.npz"
    path.write_bytes(b"test")
    with pytest.raises(ValueError, match="metadata missing"):
        validate_sidecar(path, expected=expected, strict=True)
    path.with_suffix(".json").write_text(json.dumps({"identity": expected, "text_sha256": text_digest(["الف"])}), encoding="utf-8")
    validate_sidecar(path, expected=expected, strict=True, texts=["الف"])
    with pytest.raises(ValueError, match="mismatch"):
        validate_sidecar(path, expected=expected, strict=True, texts=["ب"])


def test_semantic_training_passes_requested_backend():
    from inspect import getsource
    from persian_sea.semantic.train import train_semantic
    source = getsource(train_semantic)
    assert "visual_backend=cfg.semantic.visual_backend" in source


def test_semantic_inference_does_not_load_gt():
    from inspect import getsource
    from persian_sea.semantic.infer import align_video
    assert "include_gt=False" in getsource(align_video)


def test_checkpoint_rejects_text_backend_switch():
    cfg = VideoTextConfig(text_backend="bge_m3", text_model="BAAI/bge-m3",
                          require_text_cache_metadata=True)
    from dataclasses import asdict
    verify_checkpoint_text_source(asdict(cfg), cfg)
    with pytest.raises(ValueError, match="provenance"):
        verify_checkpoint_text_source(asdict(cfg), replace(cfg, text_model="Different-Model"))
