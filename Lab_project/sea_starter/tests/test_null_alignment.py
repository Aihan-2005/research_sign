import pytest

from persian_sea.alignment.dp import DPConfig, align_monotonic_dp
from persian_sea.io.subtitles import SubtitleCue
from persian_sea.types import SignSegment


def test_null_alignment_is_opt_in_and_keeps_existing_default():
    cues = [SubtitleCue(0.0, 1.0, "first"), SubtitleCue(20.0, 21.0, "second")]
    signs = [SignSegment(0.0, 1.0)]

    with pytest.raises(ValueError, match="Need signs>=cues"):
        align_monotonic_dp(cues, signs, DPConfig())

    aligned = align_monotonic_dp(
        cues, signs,
        DPConfig(allow_null_alignment=True, null_alignment_penalty=1.0),
    )
    assert [item.cue_index for item in aligned] == [0]


def test_null_alignment_can_return_no_proposals_when_no_signs_exist():
    cues = [SubtitleCue(0.0, 1.0, "unmatched")]
    assert align_monotonic_dp(
        cues, [], DPConfig(allow_null_alignment=True)
    ) == []
