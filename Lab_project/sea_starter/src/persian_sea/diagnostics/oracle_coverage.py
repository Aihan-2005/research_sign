
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from persian_sea.config import AppConfig, read_video_ids
from persian_sea.io.subtitles import find_subtitle, pair_corrected_cues, read_subtitles
from persian_sea.semantic.candidates import (
    enumerate_candidate_spans, interval_iou, select_candidate_indices,
)
from persian_sea.video_text.config import VideoTextConfig


THRESHOLDS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class CandidatePolicy:


    name: str
    margin_seconds: float
    min_segments: int
    max_segments: int
    span_sizes: tuple[int, ...] = ()
    max_span_segments: int | None = None
    max_span_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.name not in ("semantic", "video_text"):
            raise ValueError("policy.name must be semantic or video_text")
        if not (math.isfinite(self.margin_seconds) and self.margin_seconds >= 0):
            raise ValueError("margin must be nonnegative and finite")
        if not (1 <= self.min_segments <= self.max_segments):
            raise ValueError("invalid candidate segment limits")
        if self.name == "semantic":
            if (self.max_span_segments is None or self.max_span_segments < 1
                    or self.max_span_seconds is None or self.max_span_seconds <= 0):
                raise ValueError("semantic policy needs valid span limits")
        elif not self.span_sizes or any(size < 1 for size in self.span_sizes):
            raise ValueError("video_text policy needs positive span_sizes")

    @classmethod
    def semantic(cls, cfg: AppConfig) -> CandidatePolicy:
        s = cfg.semantic
        return cls("semantic", s.search_radius_seconds,
                   s.min_candidate_segments, s.max_candidate_segments,
                   max_span_segments=s.max_span_segments,
                   max_span_seconds=s.max_span_seconds)

    @classmethod
    def video_text(cls, cfg: VideoTextConfig) -> CandidatePolicy:
        cfg.validate()
        if cfg.pair_source != "raw":
            raise ValueError("Weak-timestamp oracle requires pair_source=raw. "
                             "Approved alignment used to select candidates is not an "
                             "independent test of weak-timestamp search coverage.")
        return cls("video_text", cfg.candidate_margin_seconds,
                   cfg.min_candidate_segments, cfg.max_candidate_segments,
                   span_sizes=tuple(cfg.span_sizes))

    def candidates(self, starts: np.ndarray, ends: np.ndarray) -> list[tuple[float, float]]:
        if self.name == "semantic":
            spans = enumerate_candidate_spans(
                starts, ends, max_span_segments=self.max_span_segments,
                max_span_seconds=self.max_span_seconds,
            )

            if not spans and len(starts):
                return [(float(starts[0]), float(ends[0]))]
            return [(span.start_time, span.end_time) for span in spans]

        n = len(starts)
        return [(float(starts[i]), float(ends[i + width - 1]))
                for width in self.span_sizes if width <= n
                for i in range(n - width + 1)]


def _validate_times(starts: np.ndarray, ends: np.ndarray, video_id: str) -> None:
    if (starts.ndim != 1 or ends.ndim != 1 or not len(starts)
            or len(starts) != len(ends) or not np.isfinite(starts).all()
            or not np.isfinite(ends).all() or np.any(starts < 0)
            or np.any(ends <= starts) or np.any(np.diff(starts) < 0)):
        raise ValueError(f"Invalid visual feature timestamps: {video_id}")


def _max_iou(candidates: list[tuple[float, float]], start: float, end: float) -> float:
    return max((interval_iou(a, b, start, end) for a, b in candidates), default=0.0)


def _unconstrained_spans(starts: np.ndarray, ends: np.ndarray) -> list[tuple[float, float]]:
    return [(float(starts[i]), float(ends[j]))
            for i in range(len(starts)) for j in range(i, len(starts))]


def evaluate_oracle(
    ids_file: str | Path, *, subtitle_dir: str | Path,
    verified_gt_dir: str | Path, feature_dir: str | Path,
    policy: CandidatePolicy, require_complete_gt: bool = False,
) -> dict:


    ids = read_video_ids(Path(ids_file))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("ID file must contain unique, nonempty video IDs")
    rows: list[dict] = []
    missing_gt: list[str] = []
    unmatched = 0
    total_cues = 0
    for video_id in ids:
        raw_file = find_subtitle(Path(subtitle_dir), video_id)
        if raw_file is None:
            raise FileNotFoundError(f"Raw subtitle missing: {video_id}")
        raw = read_subtitles(raw_file)
        total_cues += len(raw)
        gt_file = find_subtitle(Path(verified_gt_dir), video_id)
        if gt_file is None:
            missing_gt.append(video_id)
            continue
        gt = read_subtitles(gt_file)
        matched = pair_corrected_cues(raw, gt)
        unmatched += sum(verified is None for verified in matched)
        if not any(verified is not None for verified in matched):
            continue
        cache_path = Path(feature_dir) / f"{video_id}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Visual feature cache missing: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as arr:
            starts = np.asarray(arr["starts"], dtype=np.float64)
            ends = np.asarray(arr["ends"], dtype=np.float64)
        _validate_times(starts, ends, video_id)
        for cue_index, (cue, verified) in enumerate(zip(raw, matched)):
            if verified is None:
                continue
            indices = select_candidate_indices(
                starts, ends, cue.start, cue.end,
                radius=policy.margin_seconds,
                min_candidates=policy.min_segments,
                max_candidates=policy.max_segments,
            )
            selection_starts, selection_ends = starts[indices], ends[indices]
            candidate_spans = policy.candidates(selection_starts, selection_ends)

            constrained = _max_iou(candidate_spans, verified.start, verified.end)
            window_only = _max_iou(_unconstrained_spans(selection_starts, selection_ends),
                                   verified.start, verified.end)
            rows.append({
                "video_id": video_id, "cue_index": cue_index, "text": cue.text,
                "raw_start": cue.start, "raw_end": cue.end,
                "verified_start": verified.start, "verified_end": verified.end,
                "raw_iou": interval_iou(cue.start, cue.end, verified.start, verified.end),
                "oracle_iou": constrained,
                "window_unrestricted_oracle_iou": window_only,
                "loss_from_span_constraints": max(0.0, window_only - constrained),
                "num_candidate_segments": len(indices),
                "num_candidate_spans": len(candidate_spans),
                "candidate_start": float(selection_starts[0]),
                "candidate_end": float(selection_ends[-1]),
            })
    if require_complete_gt and (missing_gt or unmatched):
        raise ValueError(f"Incomplete VERIFIED GT: missing videos={missing_gt}, "
                         f"unmatched cues={unmatched}. Do not use raw VTT as GT.")
    if not rows:
        raise ValueError("No matching HUMAN-VERIFIED cues. Put approved VTTs in subtitles_gt; "
                         "raw VTT is not independent ground truth.")
    values = np.array([r["oracle_iou"] for r in rows], dtype=np.float64)
    raw_values = np.array([r["raw_iou"] for r in rows], dtype=np.float64)
    max_possible = np.array([r["window_unrestricted_oracle_iou"] for r in rows])
    aggregate = {
        "verified_cues": len(rows), "total_raw_cues": total_cues,
        "unmatched_cues_in_gt_videos": unmatched,
        "missing_verified_gt_videos": missing_gt,
        "mean_oracle_iou": float(values.mean()),
        "mean_raw_iou": float(raw_values.mean()),
        "mean_window_unrestricted_iou": float(max_possible.mean()),
        "mean_num_candidate_spans": float(np.mean([r["num_candidate_spans"] for r in rows])),
    }
    for threshold in THRESHOLDS:
        aggregate[f"oracle_recall_at_iou_{threshold:.2f}"] = float(np.mean(values >= threshold))
        aggregate[f"raw_recall_at_iou_{threshold:.2f}"] = float(np.mean(raw_values >= threshold))
    return {
        "schema_version": 1, "mode": policy.name,
        "warning": "ORACLE uses HUMAN VERIFIED timestamps: evaluation-only; "
                   "do not tune on the final test split or use this report as training labels.",
        "policy": {
            "margin_seconds": policy.margin_seconds,
            "min_segments": policy.min_segments, "max_segments": policy.max_segments,
            "span_sizes": list(policy.span_sizes),
            "max_span_segments": policy.max_span_segments,
            "max_span_seconds": policy.max_span_seconds,
        },
        "summary": aggregate, "per_cue": rows,
    }
