"""A small, replaceable orchestration layer; existing extractors remain intact."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable

from persian_sea.config import AppConfig
from persian_sea.core.registry import Registry

step_registry = Registry("preprocessing step")


def _pose(cfg: AppConfig):
    from persian_sea.pose.runner import extract_pose_all
    return extract_pose_all(cfg)


def _segment(cfg: AppConfig):
    from persian_sea.segmentation.runner import segment_all
    return segment_all(cfg)


def _visual(cfg: AppConfig):
    from persian_sea.features.scanner import scan_all
    return scan_all(cfg)


def _text(cfg: AppConfig):
    from persian_sea.text.cache import scan_text_all
    return scan_text_all(cfg)


step_registry.register("pose", _pose)
step_registry.register("segment", _segment)
step_registry.register("visual", _visual)
step_registry.register("text", _text)


def run_steps(cfg: AppConfig, steps: list[str], *, ids_file: Path | None = None,
              dry_run: bool = False) -> list[str]:
    """Specify stages explicitly; unknown stages fail *before* any side effects."""
    if not steps:
        raise ValueError("At least one step required")
    if any(s not in step_registry.options for s in steps):
        raise ValueError(f"Unknown step in {steps}; available: {step_registry.options}")
    if ids_file is not None:
        ids_file = Path(ids_file)
        if not ids_file.is_file():
            raise FileNotFoundError(ids_file)
        cfg = replace(cfg, paths=replace(cfg.paths, video_ids=ids_file))
    from persian_sea.config import read_video_ids
    ids = read_video_ids(cfg.paths.video_ids)
    if not ids:
        raise ValueError(f"No video IDs in {cfg.paths.video_ids}")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video IDs")
    if not dry_run:
        from persian_sea.features.preflight import inspect_input
        violations = []
        for vid in ids:
            review = inspect_input(vid, cfg.paths.videos, cfg.paths.subtitles_raw)
            violations.extend((vid, f) for f in review["findings"] if f["severity"] == "error")
        if violations:
            raise ValueError(f"Input preflight failed for {len(violations)} issues; run persian-sea-preflight: {violations[:4]}")
        for name in steps:
            step_registry.create(name, cfg=cfg)
    return steps
