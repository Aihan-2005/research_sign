from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from persian_sea.io.subtitles import find_subtitle, read_subtitles
from persian_sea.semantic.candidates import select_candidate_indices
from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
from persian_sea.alignment.timing_policy import resolve_offset, shift_interval
from persian_sea.translation.config import TranslationConfig


@dataclass(frozen=True)
class Ref:
    video_id: str
    cue_index: int
    text: str
    start: float
    end: float


class VisualCacheLRU:
    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: Path) -> dict[str, np.ndarray]:
        key = str(path.resolve())
        if key not in self.cache:
            with np.load(path, allow_pickle=False) as archive:
                data = {
                    key: np.asarray(archive[key])
                    for key in ("pose", "rgb", "starts", "ends")
                }
            validate_cache(data, path)
            self.cache[key] = data
        self.cache.move_to_end(key)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return self.cache[key]


def validate_cache(d: dict, path: Path | str = "cache") -> None:
    pose, rgb, starts, ends = (
        d[key] for key in ("pose", "rgb", "starts", "ends")
    )
    count = len(starts)
    if (
        count == 0
        or starts.ndim != 1
        or ends.ndim != 1
        or len(ends) != count
        or pose.ndim != 4
        or pose.shape[0] != count
        or pose.shape[-2:] != (75, 4)
        or rgb.ndim != 4
        or rgb.shape[0] != count
        or pose.shape[1] != rgb.shape[1]
        or not all(np.isfinite(x).all() for x in (pose, rgb, starts, ends))
        or np.any(starts < 0)
        or np.any(ends <= starts)
        or np.any(np.diff(starts) < 0)
    ):
        raise ValueError(f"Invalid / nonfinite visual cache: {path}")


def slice_visual(d: dict, indices: np.ndarray) -> dict:
    return {
        "pose": torch.as_tensor(d["pose"][indices].astype(np.float32)),
        "rgb": torch.as_tensor(d["rgb"][indices].astype(np.float32)),
        "starts": torch.as_tensor(d["starts"][indices].astype(np.float32)),
        "ends": torch.as_tensor(d["ends"][indices].astype(np.float32)),
    }


def load_ids(path: str | Path) -> list[str]:
    path = Path(path)
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries or len(entries) != len(set(entries)):
        raise ValueError(f"Empty/duplicate video IDs: {path}")
    return entries


class TranslationCueDataset(Dataset):
    def __init__(self, cfg: TranslationConfig, ids_path: str | Path):
        self.cfg = cfg.validate()
        self.refs: list[Ref] = []
        self.cache = VisualCacheLRU()

        for video_id in load_ids(ids_path):
            source = find_subtitle(Path(cfg.subtitles_dir), video_id)
            visual = Path(cfg.visual_cache_dir) / f"{video_id}.npz"
            if source is None or not visual.is_file():
                raise FileNotFoundError(
                    f"Need VTT and visual cache for {video_id}: {source}, {visual}"
                )

            raw_cues = read_subtitles(source)
            aligned, _ = load_alignment_cues(
                video_id,
                raw_cues,
                AlignmentSourceSettings(
                    mode=cfg.pair_source,
                    approved_dir=cfg.approved_subtitles_dir,
                    sea_dir=cfg.sea_alignment_dir,
                    semantic_dir=cfg.semantic_alignment_dir,
                    max_shift_seconds=cfg.max_alignment_shift_seconds,
                    require_alignment_review=cfg.require_alignment_review,
                ),
            )


            offset = resolve_offset(
                video_id,
                cfg.pair_source,
                cfg.subtitle_to_sign_offset_seconds,
                cfg.subtitle_offsets_by_video,
            )

            for cue_index, (cue, timing) in enumerate(zip(raw_cues, aligned)):
                if cue.text.strip():
                    start, end = shift_interval(timing.start, timing.end, offset)
                    self.refs.append(
                        Ref(video_id, cue_index, cue.text, start, end)
                    )

        if not self.refs:
            raise ValueError("No nonempty Persian sentences in selected videos")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index: int) -> dict:
        ref = self.refs[index]
        path = Path(self.cfg.visual_cache_dir) / f"{ref.video_id}.npz"
        data = self.cache.get(path)
        indices = select_candidate_indices(
            data["starts"],
            data["ends"],
            ref.start,
            ref.end,
            radius=self.cfg.cue_margin_seconds,
            min_candidates=self.cfg.min_candidate_segments,
            max_candidates=self.cfg.max_candidate_segments,
        )

        if len(indices) > self.cfg.max_source_segments:
            middle = (ref.start + ref.end) * 0.5
            centers = (data["starts"][indices] + data["ends"][indices]) / 2.0
            center_index = int(np.argmin(np.abs(centers - middle)))
            lo = max(
                0,
                min(
                    center_index - self.cfg.max_source_segments // 2,
                    len(indices) - self.cfg.max_source_segments,
                ),
            )
            indices = indices[lo:lo + self.cfg.max_source_segments]

        return {
            **slice_visual(data, indices),
            "text": ref.text,
            "video_id": ref.video_id,
            "cue_index": ref.cue_index,
        }


def inference_windows(d: dict, window_size: int, stride: int) -> list[np.ndarray]:
    if not (1 <= stride <= window_size):
        raise ValueError("Require 1 <= stride <= window_size")
    count = len(d["starts"])
    if count == 0:
        raise ValueError("Cannot decode an empty video")
    return [
        np.arange(start, min(count, start + window_size), dtype=np.int64)
        for start in range(0, count, stride)
    ]


def collate(
    batch: list[dict],
    tokenizer,
    max_target_tokens: int,
    include_targets: bool = True,
    translation_backend: str = "mt5",
) -> dict:
    if not batch:
        raise ValueError("Empty batch")
    batch_size = len(batch)
    max_segments = max(len(row["starts"]) for row in batch)
    frames, joints, channels = batch[0]["pose"].shape[1:]
    _, views, dim = batch[0]["rgb"].shape[1:]

    pose = torch.zeros(
        batch_size, max_segments, frames, joints, channels, dtype=torch.float32
    )
    rgb = torch.zeros(
        batch_size, max_segments, frames, views, dim, dtype=torch.float32
    )
    starts = torch.zeros(batch_size, max_segments, dtype=torch.float32)
    ends = torch.zeros(batch_size, max_segments, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_segments, dtype=torch.bool)

    for i, row in enumerate(batch):
        count = len(row["starts"])
        if (
            row["pose"].shape[1:] != (frames, joints, channels)
            or row["rgb"].shape[1:] != (frames, views, dim)
        ):
            raise ValueError("Inconsistent visual dimensions across examples")
        pose[i, :count], rgb[i, :count] = row["pose"], row["rgb"]
        starts[i, :count], ends[i, :count] = row["starts"], row["ends"]
        mask[i, :count] = True

    output = {
        "pose": pose, "rgb": rgb, "starts": starts, "ends": ends, "mask": mask
    }
    if include_targets:
        texts = [row["text"] for row in batch]
        if translation_backend == "mbart50":
            encoded = tokenizer(
                text_target=texts,
                padding=True,
                truncation=True,
                max_length=max_target_tokens,
                return_tensors="pt",
            )
        else:
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_target_tokens,
                return_tensors="pt",
            )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        output["labels"] = labels
        output["text"] = texts
    return output
