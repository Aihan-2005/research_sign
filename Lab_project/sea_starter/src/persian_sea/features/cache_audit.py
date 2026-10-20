

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from persian_sea.features.preflight import Finding
from persian_sea.io.subtitles import find_subtitle, read_subtitles


def inspect_cache_pair(video_id: str, visual_dir: Path, text_dir: Path,
                       subtitle_dir: Path, min_candidate_overlap: float = 0.10,
                       expected_rgb_model: str | None = None,
                       expected_text_model: str | None = None,
                       expected_samples: int | None = None,
                       expected_views: list[str] | None = None) -> dict:
    findings: list[Finding] = []
    visual_path = Path(visual_dir) / f"{video_id}.npz"
    text_path = Path(text_dir) / f"{video_id}.npz"
    sub_path = find_subtitle(Path(subtitle_dir), video_id)
    for label, path in (("visual", visual_path), ("text", text_path), ("subtitle", sub_path)):
        if path is None or not path.is_file():
            findings.append(Finding("error", f"missing_{label}", str(path)))
    if any(f.severity == "error" for f in findings):
        return {"video_id": video_id, "findings": [asdict(x) for x in findings]}
    report = {"video_id": video_id}
    try:
        cues = read_subtitles(sub_path)
        with np.load(visual_path, allow_pickle=False) as vc, np.load(text_path, allow_pickle=False) as tc:
            pose, rgb, starts, ends = (vc[k] for k in ("pose", "rgb", "starts", "ends"))
            texts, embeddings = tc["texts"].tolist(), tc["embeddings"]
            n = len(starts)
            report.update(num_signs=n, num_cues=len(cues), pose_shape=list(pose.shape),
                          rgb_shape=list(rgb.shape), text_shape=list(embeddings.shape))
            if n == 0 or pose.ndim != 4 or rgb.ndim != 4 or pose.shape[:2] != rgb.shape[:2] or pose.shape[-2:] != (75, 4) or pose.shape[0] != n or rgb.shape[0] != n or len(ends) != n:
                findings.append(Finding("error", "visual_shape", f"pose={pose.shape}, rgb={rgb.shape}, start={starts.shape}"))
            if embeddings.ndim != 2 or len(texts) != len(cues) or embeddings.shape[0] != len(cues):
                findings.append(Finding("error", "text_shape", f"embeddings={embeddings.shape}, cues={len(cues)}"))
            if list(map(str, texts)) != [c.text for c in cues]:
                findings.append(Finding("error", "stale_text_cache", "VTT text and BGE-M3 cache have different order/content"))
            if not (np.isfinite(starts).all() and np.isfinite(ends).all()) or np.any(ends <= starts) or np.any(np.diff(starts) < 0):
                findings.append(Finding("error", "sign_timestamps", "Segment time bounds invalid or unordered"))
            for name, arr in (("pose", pose), ("rgb", rgb), ("text", embeddings)):
                if not np.isfinite(arr).all():
                    findings.append(Finding("error", "nonfinite_feature", f"{name} contains NaN or Inf"))
            if pose.ndim == 4 and pose.shape[-1] == 4 and pose.size:
                missing_fraction = float((pose[..., 3] <= 0.01).mean())
                report["pose_missing_joint_fraction"] = round(missing_fraction, 4)
                if missing_fraction > 0.80:
                    findings.append(Finding("warning", "pose_low_visibility", "Most joints lack confidence: check signer ROI and MediaPipe tracking"))
            if n and len(cues) and not any(f.severity == "error" for f in findings):

                intersections = np.array([np.maximum(0., np.minimum(ends, cue.end) - np.maximum(starts, cue.start)).sum() for cue in cues])
                durations = np.array([cue.end - cue.start for cue in cues])
                fractions = intersections / np.maximum(durations, 1e-6)
                weak = int((fractions < min_candidate_overlap).sum())
                report["low_rough_overlap_cues"] = weak
                report["mean_rough_overlap_fraction"] = float(fractions.mean())
                if weak:
                    findings.append(Finding("warning", "low_rough_overlap", f"{weak}/{len(cues)} cues have low sign-segment overlap with rough times"))
    except (KeyError, ValueError, OSError, TypeError) as exc:
        findings.append(Finding("error", "cache_read_failed", str(exc)))

    visual_meta = Path(visual_dir) / f"{video_id}.json"
    text_meta = Path(text_dir) / f"{video_id}.json"
    for sidecar, name, expected, key in (
        (visual_meta, "rgb", expected_rgb_model, "rgb_model"),
        (text_meta, "text", expected_text_model, "text_model"),
    ):
        if expected is None:
            continue
        if not sidecar.is_file():
            findings.append(Finding("warning", "missing_cache_metadata", f"{name}: {sidecar}"))
        else:
            try:
                actual = json.loads(sidecar.read_text(encoding="utf-8")).get(key)
                if actual != expected:
                    findings.append(Finding("error", "stale_model_cache", f"{name}: got {actual!r}, expected {expected!r}"))
            except (OSError, ValueError) as exc:
                findings.append(Finding("error", "cache_metadata_invalid", str(exc)))
    if expected_views is not None and visual_path.is_file():
        try:
            with np.load(visual_path, allow_pickle=False) as vc:
                if "views" not in vc or list(map(str, vc["views"].tolist())) != list(expected_views):
                    findings.append(Finding("error", "stale_view_cache", "Feature cache view set/order differs"))
                if expected_samples is not None and (vc["pose"].shape[1] != expected_samples):
                    findings.append(Finding("error", "stale_sampling_cache", "Feature cache sample count differs"))
        except (KeyError, ValueError, OSError) as exc:
            findings.append(Finding("error", "cache_config_read_failed", str(exc)))
    report["findings"] = [asdict(x) for x in findings]
    return report
