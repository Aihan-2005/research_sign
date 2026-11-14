from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytest

from persian_sea.video_text.config import VideoTextConfig
from persian_sea.video_text.dataset import VideoTextDataset, collate_video_text
from persian_sea.io.subtitles import SubtitleCue, write_vtt
from persian_sea.text.cache_identity import identity_digest, text_digest
from persian_sea.video_text.losses import (
    partial_ot_similarity_matrix,
)
from persian_sea.video_text.model import VideoTextModel
from persian_sea.video_text.train import _training_objective


class ToyVisual(nn.Module):
    def __init__(self, output_dim=8, rgb_input_dim=8, **kwargs):
        super().__init__()
        self.fc = nn.Linear(rgb_input_dim + 4, output_dim)

    def forward(self, pose, rgb):
        pose_mean = pose.mean(dim=(1, 2))
        rgb_mean = rgb.mean(dim=(1, 2))
        return {
            "fused": F.normalize(
                self.fc(torch.cat([pose_mean, rgb_mean], dim=-1)), dim=-1
            )
        }


def word_config(mode="word"):
    return VideoTextConfig(
        text_representation=mode,
        text_backend="bge_m3",
        text_model="BAAI/bge-m3",
        require_text_cache_metadata=True,
        embedding_dim=8,
        model_dim=8,
        temporal_layers=0,
        temporal_heads=2,
        span_sizes=(1, 2),
        visual_microbatch=2,
        batch_size=2,
        minimum_videos_per_batch=2,
        amp=False,
    ).validate()


def toy_batch():
    batch_size, segments, words = 2, 4, 3
    return {
        "pose": torch.randn(batch_size, segments, 2, 75, 4),
        "rgb": torch.randn(batch_size, segments, 2, 4, 8),
        "starts": torch.tensor([[0., 1., 2., 3.], [0., 1., 2., 3.]]),
        "ends": torch.tensor([[.8, 1.8, 2.8, 3.8], [.8, 1.8, 2.8, 3.8]]),
        "mask": torch.ones(batch_size, segments, dtype=torch.bool),
        "text_embedding": torch.randn(batch_size, 6),
        "word_embeddings": torch.randn(batch_size, words, 6),
        "word_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "video_id": ["video-a", "video-b"],
        "text_key": ["متن الف", "متن ب"],
    }


def test_word_modes_require_bge_and_strict_cache_metadata():
    with pytest.raises(ValueError, match="bge_m3"):
        VideoTextConfig(
            text_representation="word",
            text_backend="qwen3_embedding",
            text_model="Qwen/Qwen3-Embedding-0.6B",
            require_text_cache_metadata=True,
        ).validate()
    with pytest.raises(ValueError, match="require_text_cache_metadata"):
        VideoTextConfig(
            text_representation="both",
            text_backend="bge_m3",
            text_model="BAAI/bge-m3",
            require_text_cache_metadata=False,
        ).validate()


def test_partial_ot_prefers_matching_word_video_pairs_and_has_gradients():
    words = torch.tensor(
        [[[1., 0.], [0., 1.]]], requires_grad=True
    )
    matching = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    mismatching = torch.tensor([[[-1., 0.], [0., -1.]]], requires_grad=True)
    word_mask = torch.tensor([[True, True]])
    span_mask = torch.tensor([[True, True]])

    good = partial_ot_similarity_matrix(
        words, word_mask, matching, span_mask,
        epsilon=0.05, iterations=40, dustbin_score=0.0,
    )
    bad = partial_ot_similarity_matrix(
        words, word_mask, mismatching, span_mask,
        epsilon=0.05, iterations=40, dustbin_score=0.0,
    )
    assert good.shape == (1, 1)
    assert good.item() > bad.item()
    good.sum().backward()
    assert words.grad is not None
    assert matching.grad is not None


@pytest.mark.parametrize("mode", ["word", "both"])
def test_model_forward_exposes_requested_text_representations(mode):
    cfg = word_config(mode)
    model = VideoTextModel(
        cfg, text_dim=6, rgb_dim=8, num_views=4, visual=ToyVisual()
    )
    output = model(toy_batch())
    assert output["span_vectors"].shape[0] == 2
    assert output["word_vectors"].shape[:2] == (2, 3)
    assert output["word_mask"].shape == (2, 3)
    if mode == "both":
        assert output["text_vectors"].shape == (2, 8)
    else:
        assert "text_vectors" not in output


@pytest.mark.parametrize("mode", ["word", "both"])
def test_training_objective_runs_word_partial_ot(mode):
    cfg = word_config(mode)
    model = VideoTextModel(
        cfg, text_dim=6, rgb_dim=8, num_views=4, visual=ToyVisual()
    )
    batch = toy_batch()
    output = model(batch)
    loss, info, components = _training_objective(output, batch, cfg)
    assert torch.isfinite(loss)
    assert info["active_queries"] == 2
    assert "word" in components
    if mode == "both":
        assert "sentence" in components
    loss.backward()
    assert model.word_text.network[1].weight.grad is not None


def test_sentence_only_model_does_not_create_word_tower():
    cfg = VideoTextConfig(
        text_representation="sentence",
        embedding_dim=8,
        model_dim=8,
        temporal_layers=0,
        temporal_heads=2,
        span_sizes=(1, 2),
    ).validate()
    model = VideoTextModel(
        cfg, text_dim=6, rgb_dim=8, num_views=4, visual=ToyVisual()
    )
    assert model.word_text is None
    batch = toy_batch()
    batch.pop("word_embeddings")
    batch.pop("word_mask")
    output = model(batch)
    assert output["text_vectors"].shape == (2, 8)
    assert "word_vectors" not in output



def test_dataset_loads_and_pads_ragged_word_cache(tmp_path):
    for folder in ("subs", "visual", "text", "approved"):
        (tmp_path / folder).mkdir()
    cues = [
        SubtitleCue(1.0, 2.0, "سلام دنیا"),
        SubtitleCue(3.0, 4.0, "صبح خیلی خوب"),
    ]
    write_vtt(cues, tmp_path / "subs" / "video-0.vtt")
    ids_file = tmp_path / "train_ids.txt"
    ids_file.write_text("video-0\n", encoding="utf-8")

    rng = np.random.default_rng(7)
    np.savez_compressed(
        tmp_path / "visual" / "video-0.npz",
        pose=rng.normal(size=(4, 2, 75, 4)).astype(np.float16),
        rgb=rng.normal(size=(4, 2, 4, 8)).astype(np.float16),
        starts=np.arange(4, dtype=np.float32),
        ends=np.arange(4, dtype=np.float32) + 0.8,
    )

    texts = [cue.text for cue in cues]
    identity = {
        "backend": "bge_m3",
        "model_name": "BAAI/bge-m3",
        "instruction": "",
        "model_revision": "main",
        "max_length": 256,
        "word_pooling": "mean_subword_l2",
        "word_offset_schema": 1,
    }
    np.savez_compressed(
        tmp_path / "text" / "video-0.npz",
        embeddings=rng.normal(size=(2, 6)).astype(np.float16),
        texts=np.asarray(texts),
        word_embeddings=rng.normal(size=(5, 6)).astype(np.float16),
        word_cue_offsets=np.asarray([0, 2, 5], dtype=np.int64),
        word_texts=np.asarray(["سلام", "دنیا", "صبح", "خیلی", "خوب"]),
        word_char_offsets=np.asarray(
            [[0, 4], [5, 9], [0, 3], [4, 8], [9, 12]], dtype=np.int32
        ),
        cache_identity_sha256=np.asarray(identity_digest(identity)),
    )
    sidecar = {
        "identity": identity,
        "text_sha256": text_digest(texts),
    }
    (tmp_path / "text" / "video-0.json").write_text(
        json.dumps(sidecar, ensure_ascii=False), encoding="utf-8"
    )

    cfg = VideoTextConfig(
        train_ids=str(ids_file),
        val_ids=str(ids_file),
        test_ids=str(ids_file),
        subtitles_dir=str(tmp_path / "subs"),
        approved_subtitles_dir=str(tmp_path / "approved"),
        feature_cache_dir=str(tmp_path / "visual"),
        text_cache_dir=str(tmp_path / "text"),
        output_dir=str(tmp_path / "output"),
        text_representation="word",
        text_backend="bge_m3",
        text_model="BAAI/bge-m3",
        text_revision="main",
        text_max_length=256,
        require_text_cache_metadata=True,
        embedding_dim=8,
        model_dim=8,
        temporal_layers=0,
        temporal_heads=2,
        batch_size=2,
        min_candidate_segments=2,
        max_candidate_segments=4,
    ).validate()

    dataset = VideoTextDataset(cfg, ids_file)
    first, second = dataset[0], dataset[1]
    assert first["word_embeddings"].shape == (2, 6)
    assert second["word_embeddings"].shape == (3, 6)

    batch = collate_video_text([first, second])
    assert batch["word_embeddings"].shape == (2, 3, 6)
    assert batch["word_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]
