from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from persian_sea.io.subtitles import SubtitleCue

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_subtitle_lengths.py"
spec = spec_from_file_location("phase1_text_audit", SCRIPT)
audit = module_from_spec(spec)
spec.loader.exec_module(audit)


class ToyTokenizer:
    def __call__(self, text=None, *, text_target=None, **kwargs):
        words = (text if text is not None else text_target).split()
        return {"input_ids": [1] + list(range(2, 2 + len(words))) + [3]}


def test_exact_full_token_lengths_and_truncation_detection():
    cues = [SubtitleCue(1.0, 4.0, "a b c d"), SubtitleCue(5.0, 6.0, "a")]
    rows = audit.audit_cues("v", cues, ToyTokenizer(), ToyTokenizer(), 5, 4)
    assert rows[0]["bge_tokens_full"] == 6
    assert rows[0]["bge_would_truncate"] is True
    assert rows[0]["target_would_truncate"] is True
    assert rows[1]["bge_would_truncate"] is False
