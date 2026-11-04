from __future__ import annotations

from pathlib import Path
import numpy as np

from persian_sea.io.subtitles import SubtitleCue, write_vtt
from persian_sea.video_text.config import VideoTextConfig
from persian_sea.video_text.dataset import VideoTextDataset
from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.data import TranslationCueDataset


def test_raw_clock_consistent_between_vlp_and_translation(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    visual_dir = tmp_path / "visual"
    text_dir = tmp_path / "text"
    for folder in (raw_dir, visual_dir, text_dir):
        folder.mkdir()
    write_vtt([SubtitleCue(4.0, 8.0, "سلام دنیا")], raw_dir / "v.vtt")
    starts = np.arange(12, dtype=np.float32)
    np.savez_compressed(visual_dir / "v.npz",
        pose=np.ones((12, 2, 75, 4), dtype=np.float16),
        rgb=np.ones((12, 2, 4, 8), dtype=np.float16),
        starts=starts, ends=starts+0.5)
    np.savez_compressed(text_dir / "v.npz",
        texts=np.array(["سلام دنیا"]), embeddings=np.ones((1, 8), dtype=np.float16))
    ids = tmp_path / "ids.txt"
    ids.write_text("v\n", encoding="utf-8")
    vlp = VideoTextConfig(
        subtitles_dir=str(raw_dir), feature_cache_dir=str(visual_dir),
        text_cache_dir=str(text_dir), require_text_cache_metadata=False,
        batch_size=2, subtitle_to_sign_offset_seconds=2.0)
    translation = TranslationConfig(
        subtitles_dir=str(raw_dir), visual_cache_dir=str(visual_dir),
        subtitle_to_sign_offset_seconds=2.0)
    a = VideoTextDataset(vlp, ids).refs[0]
    b = TranslationCueDataset(translation, ids).refs[0]
    assert (a.matching_start, a.matching_end) == (b.start, b.end) == (6.0, 10.0)
    assert (a.weak_start, a.weak_end) == (4.0, 8.0)
