from pathlib import Path
import numpy as np
import pytest

from persian_sea.io.subtitles import SubtitleCue, write_vtt
from persian_sea.video_text.config import VideoTextConfig
from persian_sea.video_text.dataset import VideoTextDataset
from persian_sea.translation.config import TranslationConfig
from persian_sea.translation.data import TranslationCueDataset
from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues


def fake_files(tmp_path):
    for folder in ('raw','visual','text','sea'):
        (tmp_path/folder).mkdir()
    write_vtt([SubtitleCue(3,5,'سلام')], tmp_path/'raw/v.vtt')
    write_vtt([SubtitleCue(7,9,'سلام')], tmp_path/'sea/v.proposed.vtt')
    starts = np.arange(16,dtype=np.float32)
    np.savez_compressed(tmp_path/'visual/v.npz',
                        pose=np.ones((16,2,75,4),dtype=np.float16),
                        rgb=np.ones((16,2,4,8),dtype=np.float16),
                        starts=starts,ends=starts+.8)
    np.savez_compressed(tmp_path/'text/v.npz',
                        texts=np.array(['سلام']),embeddings=np.ones((1,8),dtype=np.float16))
    (tmp_path/'ids.txt').write_text('v\n',encoding='utf8')


def test_phase1_offset_really_reaches_translation(tmp_path):
    fake_files(tmp_path)
    v = VideoTextConfig(subtitles_dir=str(tmp_path/'raw'),
                        feature_cache_dir=str(tmp_path/'visual'),
                        text_cache_dir=str(tmp_path/'text'),
                        require_text_cache_metadata=False, batch_size=2,
                        subtitle_to_sign_offset_seconds=2.0)
    t = TranslationConfig(subtitles_dir=str(tmp_path/'raw'),
                          visual_cache_dir=str(tmp_path/'visual'),
                          subtitle_to_sign_offset_seconds=2.0)
    a = VideoTextDataset(v,tmp_path/'ids.txt').refs[0]
    b = TranslationCueDataset(t,tmp_path/'ids.txt').refs[0]
    assert (a.matching_start,a.matching_end) == (b.start,b.end) == (5.,7.)


def test_corrected_pair_has_no_second_offset(tmp_path):
    fake_files(tmp_path)
    t = TranslationConfig(subtitles_dir=str(tmp_path/'raw'),
                          visual_cache_dir=str(tmp_path/'visual'),
                          pair_source='sea', sea_alignment_dir=str(tmp_path/'sea'),
                          subtitle_to_sign_offset_seconds=2.0)
    a = TranslationCueDataset(t,tmp_path/'ids.txt').refs[0]
    assert (a.start,a.end) == (7.,9.)


def test_review_gate_fails_closed_in_dataset(tmp_path):
    fake_files(tmp_path)
    t = TranslationConfig(subtitles_dir=str(tmp_path/'raw'),
                          visual_cache_dir=str(tmp_path/'visual'),
                          pair_source='sea', sea_alignment_dir=str(tmp_path/'sea'),
                          require_alignment_review=True)
    with pytest.raises(ValueError, match='Missing alignment review'):
        TranslationCueDataset(t,tmp_path/'ids.txt')
    v = VideoTextConfig(subtitles_dir=str(tmp_path/'raw'),
                        feature_cache_dir=str(tmp_path/'visual'),
                        text_cache_dir=str(tmp_path/'text'),
                        batch_size=2, pair_source='sea',
                        sea_alignment_dir=str(tmp_path/'sea'),
                        require_alignment_review=True)
    with pytest.raises(ValueError, match='Missing alignment review'):
        VideoTextDataset(v,tmp_path/'ids.txt')


def test_boundary_labels_use_phase2_timebase(tmp_path, monkeypatch):
    import persian_sea.translation.boundary as boundary
    fake_files(tmp_path)
    starts = np.arange(16, dtype=np.float32)
    features = {"features": np.zeros((16,8),dtype=np.float32),
                "starts": starts, "ends": starts+.8}
    monkeypatch.setattr(boundary, 'extract_segment_features',
                        lambda *args, **kwargs: features)
    cfg = boundary.BoundaryConfig(
        subtitles_dir=str(tmp_path/'raw'),
        visual_cache_dir=str(tmp_path/'visual'),
        pair_source='raw', subtitle_to_sign_offset_seconds=2.,
    )
    row = boundary._load_record('v', cfg, None, 'fakehash', None, extract=True)
    assert np.flatnonzero(row['labels'][:,0]).tolist() == [5]
    assert np.flatnonzero(row['labels'][:,1]).tolist() == [7]
