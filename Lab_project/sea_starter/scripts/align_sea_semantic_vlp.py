"""SEA-style semantic sentence alignment using an *already trained* VLP checkpoint.

Run from Lab_project/sea_starter after installing the project (`pip install -e .`).
This is an experimental proposal generator, never human ground truth.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from persian_sea.alignment.semantic_dp import SemanticDPSettings, align_semantic_dp
from persian_sea.alignment.quality_gate import QualityGateConfig, reconcile
from persian_sea.alignment.timing_policy import shift_cues
from persian_sea.core.contracts import AlignmentProposal, TimeSpan
from persian_sea.io.subtitles import SubtitleCue, find_subtitle, read_subtitles, write_vtt
from persian_sea.translation.boundary import load_vlp_model
from persian_sea.translation.data import validate_cache
from persian_sea.text.cache_identity import identity_digest, text_digest
from persian_sea.types import SignSegment


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pair_arrays(video_id: str, visual_path: Path, text_path: Path,
                     subtitles: list[SubtitleCue], checkpoint_cfg: dict,
                     *, allow_unverified_text_cache: bool = False) -> tuple[dict, np.ndarray]:
    """Check the exact text order and checkpoint text encoder provenance."""
    with np.load(visual_path, allow_pickle=False) as f:
        visual = {name: np.asarray(f[name]) for name in ("pose", "rgb", "starts", "ends")}
    validate_cache(visual, visual_path)
    with np.load(text_path, allow_pickle=False) as f:
        embedding = np.asarray(f["embeddings"], dtype=np.float32)
        texts = [str(x) for x in f["texts"].tolist()]
        stored_identity = str(f["cache_identity_sha256"].item()) if "cache_identity_sha256" in f else None
    expected_texts = [c.text for c in subtitles]
    if texts != expected_texts or embedding.ndim != 2 or len(embedding) != len(subtitles):
        raise ValueError(f"{video_id}: text cache mismatch with raw VTT cue order/content")
    if not np.isfinite(embedding).all():
        raise ValueError("Nonfinite text vectors")
    meta_path = text_path.with_suffix(".json")
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    identity = meta.get("identity") if isinstance(meta, dict) else None
    if isinstance(identity, dict):
        if stored_identity != identity_digest(identity):
            raise ValueError(f"{video_id}: text NPZ/sidecar identity hash mismatch")
        if meta.get("text_sha256") != text_digest(texts):
            raise ValueError(f"{video_id}: text cache text digest is stale")
    for key, field in (("backend", "text_backend"), ("model_name", "text_model"),
                       ("instruction", "text_instruction")):
        expected = checkpoint_cfg.get(field, "")
        if expected and (not isinstance(identity, dict) or identity.get(key) != expected):
            raise ValueError(f"{video_id}: text cache metadata {key} does not match VLP checkpoint")
    if not isinstance(identity, dict) and not allow_unverified_text_cache:
        raise ValueError("Text sidecar has no verifiable identity; pass --allow-unverified-text-cache only for diagnostics")
    if not checkpoint_cfg.get("text_model") and not allow_unverified_text_cache:
        raise ValueError("VLP checkpoint lacks text model identity; explicit override needed")
    return visual, embedding


@torch.inference_mode()
def similarity_from_vlp(model, visual: dict, text_vectors: np.ndarray,
                        device: torch.device, expected_dims: list[int]) -> np.ndarray:
    n = len(visual["starts"])
    if visual["rgb"].shape[-1] != expected_dims[1] or visual["rgb"].shape[-2] != expected_dims[2]:
        raise ValueError("RGB cache dimensions differ from VLP checkpoint")
    if text_vectors.shape[1] != expected_dims[0]:
        raise ValueError("Text embedding dimensions differ from VLP checkpoint")
    pose = torch.as_tensor(visual["pose"].astype(np.float32), device=device)[None]
    rgb = torch.as_tensor(visual["rgb"].astype(np.float32), device=device)[None]
    starts = torch.as_tensor(visual["starts"], dtype=torch.float32, device=device)[None]
    ends = torch.as_tensor(visual["ends"], dtype=torch.float32, device=device)[None]
    mask = torch.ones((1, n), dtype=torch.bool, device=device)
    # All video segments are jointly contextualized by the VLP temporal encoder.
    visual_tokens = model.encode_video({"pose": pose, "rgb": rgb,
                                        "starts": starts, "ends": ends,
                                        "mask": mask})[0]
    texts = model.text(torch.as_tensor(text_vectors, dtype=torch.float32, device=device))
    similarities = (texts @ visual_tokens.T).float().cpu().numpy()
    if not np.isfinite(similarities).all():
        raise FloatingPointError("Nonfinite semantic similarity scores")
    return np.clip(similarities, -1.0, 1.0)


def create_outputs(video_id, originals, aligned_input, result, output_dir: Path,
                   *, settings: SemanticDPSettings, ckpt_path: Path,
                   visual_path: Path, text_path: Path,
                   review_threshold: float, overwrite: bool,
                   similarity_matrix: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)
    proposed_path = output_dir / f"{video_id}.proposed.vtt"
    report_path = output_dir / f"{video_id}.review.json"
    if not overwrite and (proposed_path.exists() or report_path.exists()):
        raise FileExistsError("Existing output; pass --overwrite to regenerate BOTH VTT and review JSON")
    proposals = [AlignmentProposal(a.cue_index, TimeSpan(a.start, a.end),
                                   "sea_semantic_vlp") for a in result.proposals]
    _, decisions = reconcile(originals, proposals, QualityGateConfig(
        max_boundary_shift_seconds=review_threshold,
        auto_apply_small_shifts=False))
    by_id = {a.cue_index: a for a in result.proposals}
    has_null = bool(result.unmatched_cue_indices)
    report = {
        "schema_version": 1, "video_id": video_id, "method": "sea_semantic_vlp",
        "base_commit": "9baebcb686aaa464761a6f0dd546d9913fd856e4",
        "status": "incomplete_abstained" if has_null else "all_cues_proposed",
        "warning": ("Similarity is from a learned *bag-level* VLP checkpoint, not a "
                    "calibrated probability or verified sign-to-word alignment. "
                    "Do not use rejected/abstained cues as positive training examples."),
        "source_model_sha256": file_sha256(ckpt_path),
        "visual_cache_sha256": file_sha256(visual_path),
        "text_cache_sha256": file_sha256(text_path),
        "dp_settings": asdict(settings),
        "num_cues": len(originals),
        "num_unmatched_cues": len(result.unmatched_cue_indices),
        "unmatched_cue_indices": list(result.unmatched_cue_indices),
        "candidate_counts": list(result.candidate_counts),
        "dp_cost": result.total_cost, "skipped_sign_segments": result.skipped_signs,
        "proposed_vtt": str(proposed_path) if not has_null else None,
        "decisions": [
            {**asdict(d),
             "candidate": ({"start": d.proposed.start, "end": d.proposed.end}
                           if d.proposed is not None else None)}
            for d in decisions
        ],
        "proposals": [{**asdict(a),
                       "mean_cosine_similarity": float(np.mean(
                           similarity_matrix[a.cue_index, a.first_sign_index:a.last_sign_index + 1]))}
                      for a in result.proposals],
        "time_reference": "original VTT; pre-shift was used only for scoring/candidate retrieval",
    }
    # Fail closed: if ANY cue is null, do not emit a proposed.vtt with silently
    # copied raw timestamps. The existing source selector then refuses to train.
    if has_null:
        if proposed_path.is_file():
            proposed_path.unlink()  # Remove stale positive-pair source when re-running.
    else:
        proposed = [SubtitleCue(by_id[i].start, by_id[i].end, cue.text)
                    for i, cue in enumerate(originals)]
        write_vtt(proposed, proposed_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return proposed_path if not has_null else None, report_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id", required=True)
    p.add_argument("--vlp-checkpoint", required=True)
    p.add_argument("--raw-dir", default="data/subtitles_raw")
    p.add_argument("--visual-dir", default="outputs/features")
    p.add_argument("--text-dir", default="outputs/text_features/bge_m3")
    p.add_argument("--output-dir", default="outputs/aligned_subtitles/sea_semantic_vlp")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--pre-shift-seconds", type=float, default=0.0)
    p.add_argument("--search-radius", type=float, default=8.0)
    p.add_argument("--max-group-segments", type=int, default=56)
    p.add_argument("--max-group-seconds", type=float, default=22.0)
    p.add_argument("--max-internal-gap", type=float, default=6.0)
    p.add_argument("--semantic-weight", type=float, default=8.0)
    p.add_argument("--duration-weight", type=float, default=1.0)
    p.add_argument("--gap-weight", type=float, default=2.0)
    p.add_argument("--skip-sign-cost", type=float, default=0.1)
    p.add_argument("--null-cue-cost", type=float, default=14.0)
    p.add_argument("--disable-null", action="store_true")
    p.add_argument("--review-threshold", type=float, default=1.5)
    p.add_argument("--allow-unverified-text-cache", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    video_id = args.video_id
    subtitle = find_subtitle(Path(args.raw_dir), video_id)
    if subtitle is None:
        p.error(f"Raw VTT/SRT missing: {video_id}")
    originals = read_subtitles(subtitle)
    if not originals:
        raise ValueError("Empty subtitles")
    aligned_input = shift_cues(originals, args.pre_shift_seconds)
    visual_path = Path(args.visual_dir) / f"{video_id}.npz"
    text_path = Path(args.text_dir) / f"{video_id}.npz"
    checkpoint = Path(args.vlp_checkpoint)
    for path in (visual_path, text_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    vlp, payload, _ = load_vlp_model(checkpoint, device)
    visual, text = load_pair_arrays(video_id, visual_path, text_path, originals,
                                    payload["config"],
                                    allow_unverified_text_cache=args.allow_unverified_text_cache)
    settings = SemanticDPSettings(
        search_radius_seconds=args.search_radius,
        max_group_segments=args.max_group_segments,
        max_group_seconds=args.max_group_seconds,
        max_internal_gap_seconds=args.max_internal_gap,
        semantic_weight=args.semantic_weight,
        duration_weight=args.duration_weight,
        gap_weight=args.gap_weight,
        skipped_sign_cost=args.skip_sign_cost,
        allow_null_alignment=not args.disable_null,
        null_cue_cost=args.null_cue_cost,
    ).validate()
    sim = similarity_from_vlp(vlp, visual, text, device, [int(x) for x in payload["dims"]])
    signs = [SignSegment(float(s), float(e)) for s, e in zip(visual["starts"], visual["ends"])]
    result = align_semantic_dp(aligned_input, signs, sim, settings)
    proposed, report = create_outputs(
        video_id, originals, aligned_input, result, Path(args.output_dir),
        settings=settings, ckpt_path=checkpoint, visual_path=visual_path,
        text_path=text_path, review_threshold=args.review_threshold,
        overwrite=args.overwrite, similarity_matrix=sim,
    )
    print(f"cue count={len(originals)}; proposed={len(result.proposals)}; "
          f"abstained={len(result.unmatched_cue_indices)}; report={report}")
    print(f"proposed VTT={proposed if proposed is not None else 'NOT GENERATED (fail-closed)'}")
    print("WARNING: proposed times require validation, particularly shifts > review threshold")


if __name__ == "__main__":
    main()
