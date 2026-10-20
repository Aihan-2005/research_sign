from pathlib import Path
from dataclasses import asdict
import json
import pytest

from persian_sea.core.contracts import TimeSpan, AlignmentProposal
from persian_sea.core.registry import Registry
from persian_sea.alignment.backends import alignment_backends
from persian_sea.alignment.dp import DPConfig
from persian_sea.alignment.quality_gate import QualityGateConfig, reconcile
from persian_sea.io.subtitles import SubtitleCue, read_subtitles, write_vtt
from persian_sea.data.episodes import load_episode, audit_episode
from persian_sea.types import SignSegment


def test_registry_is_explicit_and_rejects_typo():
    r = Registry('test')
    r.register('baseline', lambda x: x * 2)
    assert r.create('baseline', x=3) == 6
    with pytest.raises(ValueError, match='Duplicate'):
        r.register('baseline', lambda: None)
    with pytest.raises(ValueError, match='available'):
        r.create('unknown')



def test_original_backend_does_not_require_segmentation():
    cues = [SubtitleCue(2, 4, 'سلام')]
    model = alignment_backends.create('original', dp_config=DPConfig())
    out = model.propose(cues, [])
    assert len(out) == 1 and out[0].span == TimeSpan(2, 4)


def test_sea_uses_contiguous_signs_with_global_times():
    cues = [SubtitleCue(10.2, 11.8, 'الف'), SubtitleCue(15.2, 16.8, 'ب')]
    signs = [SignSegment(0, 1), SignSegment(10, 12), SignSegment(15, 17)]
    model = alignment_backends.create('sea_temporal', dp_config=DPConfig(candidate_window_size=2))
    predicted = model.propose(cues, signs)
    assert len(predicted) == 2
    assert predicted[0].span.start >= 10 and predicted[1].span.start >= 15


def test_quality_gate_keeps_good_existing_times_by_default():
    original = [SubtitleCue(20, 24, 'سلام')]
    proposals = [AlignmentProposal(0, TimeSpan(20.2, 24.1), 'sea_temporal')]
    output, decisions = reconcile(original, proposals, QualityGateConfig())
    assert output[0].start == 20
    assert decisions[0].action == 'suggested_only'
    assert decisions[0].proposed.start == 20.2


def test_shift_larger_than_tolerance_is_flagged_not_applied():
    original = [SubtitleCue(20, 24, 'سلام')]
    proposals = [AlignmentProposal(0, TimeSpan(25, 29), 'sea_temporal')]
    output, decisions = reconcile(original, proposals, QualityGateConfig(auto_apply_small_shifts=True))
    assert output[0].start == 20
    assert decisions[0].action == 'review_large_shift'


def test_small_shift_applies_only_when_opted_in():
    originals = [SubtitleCue(20, 24, 'سلام')]
    proposals = [AlignmentProposal(0, TimeSpan(20.2, 24.1), 'sea_temporal')]
    result, review = reconcile(originals, proposals, QualityGateConfig(auto_apply_small_shifts=True))
    assert result[0].start == 20.2 and review[0].action == 'applied_small_shift'


def test_vtt_persian_round_trip_and_source_order(tmp_path: Path):
    file = tmp_path / 'video.vtt'
    cues = [SubtitleCue(1.5, 2.7, 'سلام دنیا'), SubtitleCue(4, 5, 'آب')]
    write_vtt(cues, file)
    assert [(x.start, x.end, x.text) for x in read_subtitles(file)] == [
        (1.5, 2.7, 'سلام دنیا'), (4., 5., 'آب')]


def test_vtt_with_cue_identifier_and_settings(tmp_path: Path):
    file = tmp_path / 'x.vtt'
    file.write_text('WEBVTT\n\nA\n00:00:01.000 --> 00:00:02.000 align:start\nسلام\n\n', encoding='utf-8')
    cues = read_subtitles(file)
    assert len(cues) == 1 and cues[0].text == 'سلام'


def test_episode_audit_uses_same_video_timeline(tmp_path: Path):
    videos = tmp_path / 'video'
    subtitles = tmp_path / 'sub'
    videos.mkdir(); subtitles.mkdir()
    (videos / 'episode.mp4').write_bytes(b'not a real video; ffprobe disabled in this test')
    write_vtt([SubtitleCue(0, 3, 'یک'), SubtitleCue(2.1, 5, 'دو')], subtitles / 'episode.vtt')
    ep = load_episode('episode', videos, subtitles, inspect_duration=False)
    issues = audit_episode(ep)
    assert ep.duration_seconds is None
    assert len(issues) == 1 and issues[0].code == 'overlap'


def test_invalid_time_span_rejected():
    with pytest.raises(ValueError):
        TimeSpan(5, 2)
