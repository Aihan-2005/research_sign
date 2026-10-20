from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from persian_sea.io.subtitles import SubtitleCue, shifted, pair_corrected_cues
from persian_sea.io.elan import read_sign_segments, write_aligned_subtitle_tier
from persian_sea.semantic.candidates import CandidateSpan, select_candidate_indices
from persian_sea.semantic.losses import mil_bag_score, contrastive_bag_loss, supervised_span_loss
from persian_sea.semantic.global_decode import AlignmentCandidate, monotonic_global_decode
from persian_sea.features.audio_activity import compute_energy_activity


def test_weak_and_verified_cues_are_not_mismatched_by_length():
    raw = [SubtitleCue(1, 2, "سلام"), SubtitleCue(3, 4, "خوبی")]
    gt = [SubtitleCue(8, 9, "خوبی"), SubtitleCue(5, 6, "سلام")]
    paired = pair_corrected_cues(raw, gt)
    assert [c.text for c in paired] == ["سلام", "خوبی"]
    assert [c.start for c in paired] == [5, 8]


def test_immutable_shift_and_vtt_boundaries():
    cue = SubtitleCue(10, 11, "سلام")
    result = shifted([cue], 2, 2)[0]
    assert result.start == 12 and cue.start == 10
    with pytest.raises(ValueError):
        shifted([cue], 5, 0)


def test_candidate_retrieval_preserves_contiguous_indices():
    starts = np.arange(30, dtype=float)
    indices = select_candidate_indices(
        starts, starts + 0.5,
        rough_start=12, rough_end=14,
        radius=4, min_candidates=8, max_candidates=16,
    )
    assert (np.diff(indices) == 1).all()
    assert len(indices) >= 8


def test_bag_normalization_removes_duplicate_count_bias():
    text = torch.tensor([1.0, 0.0])
    single = torch.tensor([[0.8, 0.6]])
    repeated = single.repeat(12, 1)
    a = mil_bag_score(text, single, 0.1)
    b = mil_bag_score(text, repeated, 0.1)
    assert torch.allclose(a, b, atol=1e-6)


def test_missing_or_uncaptured_gt_is_not_forced_positive():
    score = torch.tensor([0.1, 1.0], requires_grad=True)
    output = SimpleNamespace(
        spans=[CandidateSpan(0, 1, 0.0, 1.0), CandidateSpan(1, 2, 1.0, 2.0)],
        final_scores=score,
    )
    loss, count = supervised_span_loss([output], torch.tensor([8.0]), torch.tensor([10.0]), 0.1)
    assert count == 0 and loss.item() == 0
    assert loss.requires_grad


def test_same_video_far_cues_can_be_negative():
    text = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    samples = [
        SimpleNamespace(vectors=torch.tensor([[1., 0.]])),
        SimpleNamespace(vectors=torch.tensor([[0., 1.]])),
    ]
    loss, matrix = contrastive_bag_loss(
        text, samples, ["video1", "video1"],
        mil_temperature=0.1, contrastive_temperature=0.1,
        rough_start=torch.tensor([0.0, 40.0]),
        rough_end=torch.tensor([2.0, 42.0]),
    )
    assert loss.requires_grad and loss.item() > 0


def test_no_monotonic_path_raises_instead_of_invalid_fallback():
    candidates = [
        [AlignmentCandidate(10.0, 12.0, 2.0, 0)],
        [AlignmentCandidate(1.0, 3.0, 5.0, 0)],
    ]
    with pytest.raises(ValueError, match="No monotonic path"):
        monotonic_global_decode(candidates, [(10, 12), (15, 17)], 2.0, 0.1)


def test_elan_output_preserves_sign_tier(tmp_path: Path):
    source = tmp_path / "source.eaf"
    source.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT><TIME_ORDER>
<TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="1000" />
<TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="2000" />
</TIME_ORDER>
<TIER TIER_ID="SIGN" LINGUISTIC_TYPE_REF="default-lt"><ANNOTATION>
<ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
<ANNOTATION_VALUE>sign</ANNOTATION_VALUE>
</ALIGNABLE_ANNOTATION></ANNOTATION></TIER>
<LINGUISTIC_TYPE LINGUISTIC_TYPE_ID="default-lt" TIME_ALIGNABLE="true" />
</ANNOTATION_DOCUMENT>''', encoding="utf-8")
    target = tmp_path / "aligned.eaf"
    write_aligned_subtitle_tier(source, [SubtitleCue(1.3, 2.5, "سلام")], target)
    signs, tier = read_sign_segments(target, ["SIGN"])
    captions, caption_tier = read_sign_segments(target, ["SUBTITLE_SHIFTED"])
    assert len(signs) == 1 and signs[0].start == 1.0
    assert len(captions) == 1 and captions[0].text == "سلام"


def test_audio_activity_is_an_auxiliary_signal():
    wave = np.concatenate([np.zeros(1600), np.ones(1600) * 0.5]).astype(np.float32)
    times, rms, active = compute_energy_activity(wave, sample_rate=16000)
    assert len(times) == len(rms) == len(active)
    assert active[-1] and not active[0]
