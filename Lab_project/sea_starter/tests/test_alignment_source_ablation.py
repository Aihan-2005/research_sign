
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
from persian_sea.io.subtitles import SubtitleCue, write_vtt
from persian_sea.video_text.config import VideoTextConfig
from persian_sea.video_text.dataset import VideoTextDataset
from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.data import TranslationCueDataset, collate


RAW = [SubtitleCue(1., 2., "سلام"), SubtitleCue(4., 5., "امروز باران می بارد")]
SEA = [SubtitleCue(2., 3., "سلام"), SubtitleCue(6., 7., "امروز باران می بارد")]
SEMANTIC = [SubtitleCue(1.3, 2.3, "سلام"), SubtitleCue(5., 6., "امروز باران می بارد")]


def setup_data(root: Path) -> dict:
    dirs = {k: root / k for k in ("raw", "approved", "sea", "semantic", "visual", "text")}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    write_vtt(RAW, dirs["raw"] / "v.vtt")
    write_vtt(SEA, dirs["sea"] / "v.proposed.vtt")
    write_vtt(SEMANTIC, dirs["semantic"] / "v.proposed.vtt")
    write_vtt(SEA, dirs["approved"] / "v.vtt")
    np.savez_compressed(dirs["visual"] / "v.npz",
        pose=np.ones((12, 2, 75, 4), dtype=np.float16),
        rgb=np.ones((12, 2, 4, 8), dtype=np.float16),
        starts=np.arange(12, dtype=np.float32),
        ends=np.arange(12, dtype=np.float32) + 0.5)
    np.savez_compressed(dirs["text"] / "v.npz",
        texts=np.array([x.text for x in RAW]),
        embeddings=np.ones((2, 8), dtype=np.float16))
    ids = root / "ids.txt"
    ids.write_text("v\n", encoding="utf-8")
    return {**dirs, "ids": ids}


def settings(d, mode):
    return AlignmentSourceSettings(mode, str(d["approved"]), str(d["sea"]),
                                   str(d["semantic"]))


@pytest.mark.parametrize("mode,starts", [
    ("raw", (1., 4.)), ("approved", (2., 6.)),
    ("sea", (2., 6.)), ("semantic", (1.3, 5.))])
def test_modes_preserve_label_order_and_choose_only_times(tmp_path, mode, starts):
    d = setup_data(tmp_path)
    aligned, src = load_alignment_cues("v", RAW, settings(d, mode))
    assert tuple(x.start for x in aligned) == pytest.approx(starts)
    assert [x.text for x in aligned] == [x.text for x in RAW]
    assert (src is None) == (mode == "raw")


def test_no_fallback_when_alignment_missing(tmp_path):
    d = setup_data(tmp_path)
    (d["sea"] / "v.proposed.vtt").unlink()
    with pytest.raises(FileNotFoundError, match="no raw fallback"):
        load_alignment_cues("v", RAW, settings(d, "sea"))


def test_no_accidental_use_of_quality_gated_applied_vtt(tmp_path):
    d = setup_data(tmp_path)
    (d["sea"] / "v.proposed.vtt").unlink()
    write_vtt(RAW, d["sea"] / "v.vtt")
    with pytest.raises(FileNotFoundError, match="no raw fallback"):
        load_alignment_cues("v", RAW, settings(d, "sea"))


@pytest.mark.parametrize("broken", [
    [SubtitleCue(2., 3., "متن اشتباه"), SEA[1]],
    SEA[:1],
    [SubtitleCue(8., 9., "سلام"), SEA[1]],
])
def test_bad_alignment_fails_closed(tmp_path, broken):
    d = setup_data(tmp_path)
    write_vtt(broken, d["sea"] / "v.proposed.vtt")
    with pytest.raises(ValueError):
        load_alignment_cues("v", RAW, settings(d, "sea"))


def test_large_shift_limit(tmp_path):
    d = setup_data(tmp_path)
    cfg = replace(settings(d, "sea"), max_shift_seconds=0.5)
    with pytest.raises(ValueError, match="shifted"):
        load_alignment_cues("v", RAW, cfg)


def test_video_text_and_translation_share_identical_alignment_times(tmp_path):
    d = setup_data(tmp_path)
    vt = VideoTextConfig(subtitles_dir=str(d["raw"]), text_cache_dir=str(d["text"]),
                         feature_cache_dir=str(d["visual"]),
                         pair_source="sea", sea_alignment_dir=str(d["sea"]),
                         batch_size=2, min_candidate_segments=2, max_candidate_segments=8,
                         require_text_cache_metadata=False)
    vds = VideoTextDataset(vt, d["ids"])
    assert (vds.refs[0].weak_start, vds.refs[0].matching_start) == (1., 2.)
    assert vds.refs[1].matching_start == 6.
    tr = TranslationConfig(subtitles_dir=str(d["raw"]), visual_cache_dir=str(d["visual"]),
                           pair_source="sea", sea_alignment_dir=str(d["sea"]))
    tds = TranslationCueDataset(tr, d["ids"])
    assert [r.start for r in tds.refs] == [2., 6.]
    assert [r.text for r in tds.refs] == [r.text for r in vds.refs]


class FakeMBartTokenizer:
    def __call__(self, *args, **kwargs):
        assert not args
        assert kwargs["text_target"] == ["سلام"]
        return {"input_ids": torch.tensor([[10, 11, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0]])}


def test_mbart_target_encoding_masks_padding():
    item = {"pose": torch.ones(2, 2, 75, 4), "rgb": torch.ones(2, 2, 4, 8),
            "starts": torch.tensor([0., 1.]), "ends": torch.tensor([0.5, 1.5]),
            "text": "سلام"}
    out = collate([item], FakeMBartTokenizer(), 12, translation_backend="mbart50")
    assert out["labels"].tolist() == [[10, 11, -100]]


def test_validation_of_config_options():
    assert TranslationConfig(translation_backend="mbart50", model_name="facebook/mbart-large-50-many-to-many-mmt").validate()
    assert VideoTextConfig(pair_source="semantic").validate()
    with pytest.raises(ValueError, match="pair_source"):
        VideoTextConfig(pair_source="invented").validate()
    with pytest.raises(ValueError, match="target_lang"):
        TranslationConfig(translation_backend="mbart50", model_name="facebook/mbart-large-50-many-to-many-mmt", target_lang="en_XX").validate()


def test_mbart_generation_config_fa_ir_without_model_download():
    from types import SimpleNamespace
    from persian_sea.translation.language import configure_generation
    cfg = TranslationConfig(translation_backend="mbart50",
                            model_name="facebook/mbart-large-50-many-to-many-mmt")
    model = SimpleNamespace(generation_config=SimpleNamespace(forced_bos_token_id=None))
    tok = SimpleNamespace(lang_code_to_id={"fa_IR": 250032})
    configure_generation(model, tok, cfg)
    assert model.generation_config.forced_bos_token_id == 250032


def test_alignment_source_audit_provenance_hash(tmp_path):
    import importlib.util
    src = Path(__file__).parents[1] / "scripts/audit_alignment_sources.py"
    spec = importlib.util.spec_from_file_location("pilot_alignment_audit", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dirs = setup_data(tmp_path)
    row = module.audit_one("v", str(dirs["raw"]), settings(dirs, "sea"))
    assert row["num_cues"] == 2
    assert len(row["source_sha256"]) == 64
    assert row["source_sha256"] != row["raw_sha256"]
