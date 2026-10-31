from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json

from persian_sea.alignment.backends import alignment_backends
from persian_sea.alignment.dp import DPConfig
from persian_sea.alignment.quality_gate import QualityGateConfig, reconcile
from persian_sea.config import AppConfig, read_video_ids, segmentation_subdir
from persian_sea.core.contracts import AlignmentProposal, TimeSpan
from persian_sea.io.elan import read_sign_segments, write_aligned_subtitle_tier
from persian_sea.io.subtitles import SubtitleCue, find_subtitle, read_subtitles, shifted, write_vtt


def align_one(video_id: str, cfg: AppConfig) -> Path:
    subtitle_path = find_subtitle(cfg.paths.subtitles_raw, video_id)
    if subtitle_path is None:
        raise FileNotFoundError(f"Subtitle missing for {video_id}")
    originals = read_subtitles(subtitle_path)
    if not originals:
        raise ValueError(f"No subtitle cues for {video_id}")
    out = cfg.paths.alignment_output / f"{video_id}.vtt"
    if out.exists() and not cfg.alignment.overwrite:
        return out
    backend = cfg.alignment.backend
    signs = []
    eaf_directory = (Path(cfg.alignment.sign_eaf_dir)
                     if cfg.alignment.sign_eaf_dir else segmentation_subdir(cfg))
    source_eaf = eaf_directory / f"{video_id}.eaf"
    if backend != "original":
        if not source_eaf.exists():
            raise FileNotFoundError(f"Need SIGN segments: {source_eaf}")
        signs, _ = read_sign_segments(source_eaf, cfg.elan.sign_tiers)
    if cfg.alignment.similarity_weight != 0.0:
        raise ValueError(
            "SEA pipeline does not currently build a semantic similarity matrix. "
            "Keep alignment.similarity_weight=0 until a sim_matrix provider is connected."
        )
    dp_config = DPConfig(
        duration_penalty_weight=cfg.alignment.duration_penalty_weight,
        gap_penalty_weight=cfg.alignment.gap_penalty_weight,
        max_gap_seconds=cfg.alignment.max_gap_seconds,
        candidate_window_size=cfg.alignment.candidate_window_size,
        # Similarity remains zero until the pipeline supplies a real sim_matrix.
        similarity_weight=cfg.alignment.similarity_weight,
        allow_null_alignment=cfg.alignment.allow_null_alignment,
        null_alignment_penalty=cfg.alignment.null_alignment_penalty,
    )
    aligner = alignment_backends.create(backend, dp_config=dp_config)
    shifted_inputs = shifted(
        originals, cfg.alignment.pre_shift_start_seconds,
        cfg.alignment.pre_shift_end_seconds)
    proposals = aligner.propose(shifted_inputs, signs)

    proposals = [AlignmentProposal(
        p.cue_index, TimeSpan(
            max(0., p.span.start + cfg.alignment.post_shift_start_seconds),
            p.span.end + cfg.alignment.post_shift_end_seconds,
        ), p.method, p.score)
        for p in proposals]

    applied, decisions = reconcile(originals, proposals, QualityGateConfig(
        max_boundary_shift_seconds=cfg.alignment.max_boundary_shift_seconds,
        auto_apply_small_shifts=cfg.alignment.auto_apply_small_shifts,
    ))
    by_index = {p.cue_index: p for p in proposals}
    proposed_cues = [
        SubtitleCue(by_index[i].span.start, by_index[i].span.end, cue.text)
        if i in by_index else cue
        for i, cue in enumerate(originals)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    proposal_file = out.with_name(f"{video_id}.proposed.vtt")
    write_vtt(proposed_cues, proposal_file)
    write_vtt(applied, out)
    report_file = out.with_name(f"{video_id}.review.json")
    report_file.write_text(json.dumps({
        "video_id": video_id, "method": backend,
        "warning": "method scores are NOT calibrated probabilities / verified GT",
        "null_alignment_enabled": cfg.alignment.allow_null_alignment,
        "null_alignment_penalty": cfg.alignment.null_alignment_penalty,
        "candidate_vtt": str(proposal_file), "applied_vtt": str(out),
        "decisions": [asdict(d) for d in decisions],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if cfg.alignment.write_updated_eaf and source_eaf.exists():
        target_eaf = cfg.paths.alignment_output / f"{video_id}.predicted.eaf"
        write_aligned_subtitle_tier(source_eaf, applied, target_eaf,
                                    tier_id=cfg.elan.aligned_subtitle_tier)
    print(f"[align] {video_id}: {len(originals)} cues, {len(signs)} signs, "
          f"backend={backend}, proposals={proposal_file}, review={report_file}")
    return out


def align_all(cfg: AppConfig) -> None:
    for video_id in read_video_ids(cfg.paths.video_ids):
        align_one(video_id, cfg)
