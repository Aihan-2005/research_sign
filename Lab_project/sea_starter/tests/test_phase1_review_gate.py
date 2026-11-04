from __future__ import annotations

import json
from pathlib import Path
import pytest

from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
from persian_sea.io.subtitles import SubtitleCue, write_vtt


def setup(tmp_path: Path, actions: list[str] | None):
    raw = [SubtitleCue(10, 12, "سلام"), SubtitleCue(13, 15, "خوبی")]
    sea = [SubtitleCue(11, 13, "سلام"), SubtitleCue(14, 16, "خوبی")]
    proposed = tmp_path / "v.proposed.vtt"
    write_vtt(sea, proposed)
    if actions is not None:
        (tmp_path / "v.review.json").write_text(json.dumps({
            "decisions": [{"cue_index": i, "action": a} for i, a in enumerate(actions)]
        }), encoding="utf-8")
    strict = AlignmentSourceSettings(mode="sea", sea_dir=str(tmp_path),
                                     require_alignment_review=True)
    return raw, strict


def test_review_gate_allows_weak_but_nonrejected_proposals(tmp_path):
    raw, settings = setup(tmp_path, ["suggested_only", "unchanged"])
    selected, _ = load_alignment_cues("v", raw, settings)
    assert [(c.start, c.end) for c in selected] == [(11, 13), (14, 16)]


@pytest.mark.parametrize("blocked", ["review_order", "review_large_shift", "no_proposal"])
def test_review_gate_rejects_invalid_pairs(tmp_path, blocked):
    raw, settings = setup(tmp_path, ["suggested_only", blocked])
    with pytest.raises(ValueError, match="review gate rejected"):
        load_alignment_cues("v", raw, settings)


def test_review_gate_requires_a_report(tmp_path):
    raw, settings = setup(tmp_path, None)
    with pytest.raises(ValueError, match="Missing alignment review"):
        load_alignment_cues("v", raw, settings)


def test_review_gate_detects_incomplete_decisions(tmp_path):
    raw, settings = setup(tmp_path, ["suggested_only", "unchanged"])
    (tmp_path / "v.review.json").write_text(json.dumps({
        "decisions": [{"cue_index": 0, "action": "suggested_only"},
                      {"cue_index": 0, "action": "unchanged"}]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate/missing"):
        load_alignment_cues("v", raw, settings)


def test_approved_mode_does_not_require_sea_review(tmp_path):
    raw, settings = setup(tmp_path, None)
    write_vtt(raw, tmp_path / "v.vtt")
    approved = AlignmentSourceSettings(mode="approved", approved_dir=str(tmp_path),
                                       require_alignment_review=True)
    selected, _ = load_alignment_cues("v", raw, approved)
    assert selected == raw


def test_review_gate_detects_stale_timing(tmp_path):
    raw, settings = setup(tmp_path, ["suggested_only", "unchanged"])
    (tmp_path / "v.review.json").write_text(json.dumps({
        "decisions": [
            {"cue_index": 0, "action": "suggested_only",
             "candidate": {"start": 100.0, "end": 102.0}},
            {"cue_index": 1, "action": "unchanged"}
        ]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Stale review"):
        load_alignment_cues("v", raw, settings)
