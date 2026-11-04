from __future__ import annotations

import pytest

from persian_sea.alignment.timing_policy import (
    expected_timing, resolve_offset, shift_cues, shift_interval,
)
from persian_sea.io.subtitles import SubtitleCue


def test_shared_raw_offset_and_per_video_override():
    overrides = {"news001": 3.25}
    assert resolve_offset("news001", "raw", 2.0, overrides) == 3.25
    assert resolve_offset("other", "raw", 2.0, overrides) == 2.0
    for source in ("sea", "semantic", "approved"):
        assert resolve_offset("news001", source, 2.0, overrides) == 0.0


def test_cues_are_shifted_without_mutating_raw_times():
    raw = [SubtitleCue(40.0, 45.0, "من به دانشگاه می روم")]
    adjusted = expected_timing(raw, "news001", "raw", 2.0)
    assert (adjusted[0].start, adjusted[0].end) == (42.0, 47.0)
    assert (raw[0].start, raw[0].end) == (40.0, 45.0)
    assert expected_timing(adjusted, "news001", "sea", 2.0) == adjusted


def test_near_zero_clamp_and_negative_invalid():
    assert shift_interval(0.5, 3.0, -0.3) == pytest.approx((0.2, 2.7))
    with pytest.raises(ValueError, match="invalidates interval"):
        shift_interval(0.5, 1.0, -1.2)
    with pytest.raises(ValueError):
        resolve_offset("v", "raw", float("nan"))


def test_shift_cues_preserves_text_and_order():
    raw = [SubtitleCue(2, 3, "سلام"), SubtitleCue(4, 5, "خوبی")]
    after = shift_cues(raw, 1)
    assert [(x.start, x.text) for x in after] == [(3, "سلام"), (5, "خوبی")]
