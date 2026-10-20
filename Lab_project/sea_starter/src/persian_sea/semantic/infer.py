from __future__ import annotations

from pathlib import Path
import json
import tempfile

import torch
from torch.utils.data import DataLoader

from persian_sea.config import AppConfig
from persian_sea.io.subtitles import SubtitleCue, find_subtitle, read_subtitles, write_vtt
from persian_sea.semantic.dataset import SubtitleAlignmentDataset, collate_alignment_batch
from persian_sea.semantic.global_decode import AlignmentCandidate
from persian_sea.semantic.decoder_registry import decoders
from persian_sea.core.contracts import AlignmentProposal, TimeSpan
from persian_sea.alignment.quality_gate import QualityGateConfig, reconcile
from persian_sea.semantic.model import SemanticAligner
from persian_sea.semantic.train import move_batch, resolve_device


def load_model(cfg: AppConfig, checkpoint_path: Path):
    device = resolve_device(cfg.training.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SemanticAligner(
        text_input_dim=int(payload["text_dim"]),
        rgb_input_dim=int(payload["rgb_dim"]),
        num_views=int(payload["num_views"]),
        embedding_dim=int(payload["embedding_dim"]),
        visual_backend=str(payload.get("visual_backend", "pose_rgb")),
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), device


def _decode_with_expanding_topk(
    ranked_candidates: list[list[AlignmentCandidate]],
    rough_times: list[tuple[float, float]], cfg: AppConfig,
) -> list[AlignmentCandidate]:
    limit = max(1, cfg.semantic.global_top_k)
    max_size = max(len(group) for group in ranked_candidates)
    while True:
        selected = [group[:limit] for group in ranked_candidates]
        try:
            return decoders.create(cfg.semantic.global_decoder)(
                selected, rough_times,
                overlap_penalty=cfg.semantic.overlap_penalty,
                gap_change_penalty=cfg.semantic.gap_change_penalty,
            )
        except ValueError as exc:
            if limit >= max_size:
                raise ValueError(
                    "No valid monotonic alignment even with all candidates. "
                    "Inspect video segmentation or expand search radius."
                ) from exc
            limit = min(max_size, limit * 2)


@torch.no_grad()
def align_video(video_id: str, cfg: AppConfig,
                checkpoint_path: Path) -> Path:
    """Infer all cues from a continuous video jointly, without subtitle cropping."""
    model, device = load_model(cfg, checkpoint_path)

    with tempfile.TemporaryDirectory(prefix="persian_sea_") as tmp:
        ids_path = Path(tmp) / "video_ids.txt"
        ids_path.write_text(video_id + "\n", encoding="utf-8")
        dataset = SubtitleAlignmentDataset(cfg, ids_path, include_gt=False)
        if any(ref.video_id != video_id for ref in dataset.refs):
            raise RuntimeError("Unexpected sample video in inference")
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_alignment_batch,
        )
        ranked_candidates = []
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(
                batch,
                max_span_segments=cfg.semantic.max_span_segments,
                max_span_seconds=cfg.semantic.max_span_seconds,
                align_temperature=cfg.semantic.align_temperature,
                temporal_prior_weight=cfg.semantic.temporal_prior_weight,
                temporal_sigma_seconds=cfg.semantic.temporal_sigma_seconds,
                duration_prior_weight=cfg.semantic.duration_prior_weight,
            )
            sample = output["samples"][0]
            score_values, positions = torch.sort(sample.final_scores, descending=True)
            ranked_candidates.append([
                AlignmentCandidate(
                    start=sample.spans[index].start_time,
                    end=sample.spans[index].end_time,
                    score=float(value),
                    local_rank=rank,
                )
                for rank, (value, index) in enumerate(
                    zip(score_values.tolist(), positions.tolist())
                )
            ])

    raw_path = find_subtitle(cfg.paths.subtitles_raw, video_id)
    if raw_path is None:
        raise FileNotFoundError(f"Missing raw subtitles: {video_id}")
    rough = read_subtitles(raw_path)
    if len(rough) != len(ranked_candidates):
        raise RuntimeError("Number of predictions does not equal number of subtitles")
    selected = _decode_with_expanding_topk(
        ranked_candidates, [(c.start, c.end) for c in rough], cfg,
    )
    proposals = [
        AlignmentProposal(i, TimeSpan(s.start, s.end), method="learned_semantic", score=s.score)
        for i, s in enumerate(selected)
    ]
    applied, decisions = reconcile(
        rough, proposals, QualityGateConfig(
            max_boundary_shift_seconds=cfg.alignment.max_boundary_shift_seconds,
            auto_apply_small_shifts=cfg.alignment.auto_apply_small_shifts,
        ),
    )
    candidate_cues = [SubtitleCue(s.start, s.end, rough[i].text)
                      for i, s in enumerate(selected)]
    out_path = cfg.paths.alignment_output / "semantic" / f"{video_id}.vtt"
    proposed_path = out_path.with_name(f"{video_id}.proposed.vtt")
    write_vtt(candidate_cues, proposed_path)
    write_vtt(applied, out_path)
    debug_path = out_path.with_suffix(".review.json")
    debug_path.write_text(json.dumps({
        "video_id": video_id,
        "warning": "semantic score is not a calibrated confidence probability",
        "proposed_vtt": str(proposed_path), "applied_vtt": str(out_path),
        "decisions": [
            {"cue_index": i, "text": rough[i].text,
             "rough_start": rough[i].start, "rough_end": rough[i].end,
             "pred_start": selected[i].start, "pred_end": selected[i].end,
             "score": selected[i].score, "local_rank": selected[i].local_rank,
             "action": decisions[i].action,
             "boundary_shift_seconds": decisions[i].max_boundary_delta}
            for i in range(len(selected))
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[semantic-align] {video_id} -> {out_path}")
    return out_path

