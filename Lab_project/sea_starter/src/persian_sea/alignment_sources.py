"""Resolve weak/corrected subtitle timestamps with an opt-in, fail-closed review gate.

Legacy behavior remains the default for historical experiments. For research
training, set require_alignment_review=True in VLP/Translation/Boundary configs.
The review gate rejects questionable SEA/semantic proposals; it does NOT
certify accepted proposals as human-verified ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json

from persian_sea.io.subtitles import SubtitleCue, find_subtitle, read_subtitles

SOURCES = frozenset({"raw", "approved", "sea", "semantic"})
REJECT_REVIEW_ACTIONS = frozenset({"review_large_shift", "review_order", "no_proposal"})
EXPECTED_REVIEW_ACTIONS = frozenset({
    "review_large_shift", "review_order", "no_proposal",
    "applied_small_shift", "suggested_only", "unchanged",
})


@dataclass(frozen=True)
class AlignmentSourceSettings:
    mode: str = "raw"
    approved_dir: str = "data/subtitles_approved"
    sea_dir: str = "outputs/aligned_subtitles"
    semantic_dir: str = "outputs/aligned_subtitles/semantic"
    max_shift_seconds: float = 0.0
    require_alignment_review: bool = False

    def validate(self) -> "AlignmentSourceSettings":
        if self.mode not in SOURCES:
            raise ValueError(f"Unknown pair_source {self.mode!r}; choose {sorted(SOURCES)}")
        if not (0 <= self.max_shift_seconds < float("inf")):
            raise ValueError("max_shift_seconds must be finite and nonnegative")
        if not isinstance(self.require_alignment_review, bool):
            raise ValueError("require_alignment_review must be boolean")
        return self


def review_file_for(source: Path) -> Path:
    """SEA: <id>.review.json, semantic: <id>.review.json."""
    name = source.name
    if not name.endswith(".proposed.vtt"):
        raise ValueError(f"Not a proposed alignment VTT: {source}")
    return source.with_name(name[:-len(".proposed.vtt")] + ".review.json")


def read_review_actions(source: Path, expected_count: int, *, required: bool) -> list[str] | None:
    report_path = review_file_for(source)
    if not report_path.is_file():
        if required:
            raise ValueError(f"Missing alignment review report: {report_path}")
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Corrupt alignment review report: {report_path}") from exc
    records = report.get("decisions")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"Review cue count mismatch: {report_path}")
    actions = [""] * expected_count
    seen = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid review entry: {report_path}")
        i, action = item.get("cue_index"), item.get("action")
        if not isinstance(i, int) or isinstance(i, bool) or not (0 <= i < expected_count) or i in seen:
            raise ValueError(f"Duplicate/missing/invalid review cue index in {report_path}")
        if action not in EXPECTED_REVIEW_ACTIONS:
            raise ValueError(f"Unknown review action {action!r} in {report_path}")
        seen.add(i)
        actions[i] = action
    if len(seen) != expected_count:
        raise ValueError(f"Missing review actions: {report_path}")
    return actions


def verify_review_matches_proposal(source: Path, selected: Sequence[SubtitleCue]) -> None:
    """Catch stale SEA/semantic review JSON paired with a newer proposed VTT.

    Candidate boundaries are rounded to milliseconds by write_vtt, so 2ms
    tolerance is allowed. Reviews lacking candidate fields remain weak metadata.
    """
    report = json.loads(review_file_for(source).read_text(encoding="utf-8"))
    for row in report["decisions"]:
        i = int(row["cue_index"])
        candidate = row.get("candidate")
        if isinstance(candidate, dict):
            start, end = candidate.get("start"), candidate.get("end")
        elif "pred_start" in row and "pred_end" in row:
            start, end = row["pred_start"], row["pred_end"]
        else:
            continue
        if start is None or end is None:
            continue
        try:
            delta = max(abs(float(start)-selected[i].start),
                        abs(float(end)-selected[i].end))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid candidate timing in {review_file_for(source)}") from exc
        if delta > 0.002:
            raise ValueError(
                f"Stale review/proposed VTT mismatch for cue {i} ({delta:.3f}s): "
                f"{source}. Regenerate alignment/review together."
            )


def load_alignment_cues(
    video_id: str, raw_cues: Sequence[SubtitleCue],
    settings: AlignmentSourceSettings,
) -> tuple[list[SubtitleCue], Path | None]:
    settings.validate()
    if not raw_cues:
        raise ValueError(f"{video_id}: empty raw subtitle list")
    if settings.mode == "raw":
        return list(raw_cues), None
    if settings.mode == "approved":
        source = find_subtitle(Path(settings.approved_dir), video_id)
    else:
        parent = Path(settings.sea_dir if settings.mode == "sea" else settings.semantic_dir)
        source = parent / f"{video_id}.proposed.vtt"
        if not source.is_file():
            source = None
    if source is None:
        raise FileNotFoundError(f"{video_id}: {settings.mode} timing file missing; no raw fallback")
    selected = read_subtitles(source)
    if len(selected) != len(raw_cues):
        raise ValueError(f"{video_id}: {settings.mode} cue count mismatch: "
                         f"{len(selected)} != {len(raw_cues)}")
    last_start = -1.0
    for i, (raw, aligned) in enumerate(zip(raw_cues, selected)):
        if raw.text != aligned.text:
            raise ValueError(f"{video_id}: {settings.mode} text/order changed at cue {i}")
        if aligned.start < last_start or aligned.start < 0 or aligned.end <= aligned.start:
            raise ValueError(f"{video_id}: invalid/non-monotonic timestamps at cue {i}")
        last_start = aligned.start
        delta = max(abs(raw.start - aligned.start), abs(raw.end - aligned.end))
        if settings.max_shift_seconds and delta > settings.max_shift_seconds:
            raise ValueError(
                f"{video_id}: {settings.mode} cue {i} shifted {delta:.3f}s "
                f"> configured maximum {settings.max_shift_seconds:.3f}s"
            )
    if settings.mode in {"sea", "semantic"}:
        actions = read_review_actions(
            source, len(raw_cues), required=settings.require_alignment_review)
        if settings.require_alignment_review:
            if actions is None:
                raise AssertionError("Required review actions unexpectedly missing")
            verify_review_matches_proposal(source, selected)
            rejected = [(i, action) for i, action in enumerate(actions)
                        if action in REJECT_REVIEW_ACTIONS]
            if rejected:
                detail = ", ".join(f"cue {i}: {action}" for i, action in rejected[:12])
                raise ValueError(
                    f"{video_id}: review gate rejected weak alignment: {detail}. "
                    "Human-approve/rewrite this video, or train using 'approved'. "
                    "Do not silently convert no_proposal to a positive pair."
                )
    return selected, source
