from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from persian_sea.config import AppConfig, read_video_ids
from persian_sea.features.cache import FeatureCache
from persian_sea.io.subtitles import SubtitleCue, find_subtitle, pair_corrected_cues, read_subtitles
from persian_sea.semantic.candidates import select_candidate_indices
from persian_sea.text.cache import TextFeatureCache, text_cache_directory, verify_text_cache


@dataclass(frozen=True)
class SampleRef:
    video_id: str
    cue_index: int
    rough: SubtitleCue
    corrected: SubtitleCue | None


class _VideoCacheLRU:
    def __init__(self, max_items: int = 2):
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.max_items = max_items
        self._visual: OrderedDict[str, FeatureCache] = OrderedDict()
        self._text: OrderedDict[str, TextFeatureCache] = OrderedDict()

    def _get(self, cache: OrderedDict, path: Path, video_id: str, constructor):
        if video_id in cache:
            cache.move_to_end(video_id)
            return cache[video_id]
        value = constructor(path)
        cache[video_id] = value
        while len(cache) > self.max_items:
            cache.popitem(last=False)
        return value

    def visual(self, path: Path, video_id: str) -> FeatureCache:
        return self._get(self._visual, path, video_id, FeatureCache)

    def text(self, path: Path, video_id: str) -> TextFeatureCache:
        return self._get(self._text, path, video_id, TextFeatureCache)


class SubtitleAlignmentDataset(Dataset):
    """Weakly aligned continuous-video cues; GT only if manually verified.

    A cue's approximate start/end are used only for candidate retrieval and
    temporal priors. Place ONLY independently verified timings in subtitles_gt.
    """
    def __init__(self, cfg: AppConfig, video_ids_path: Path,
                 memory_cache_videos: int = 2, include_gt: bool = True):
        self.cfg = cfg
        self.refs: list[SampleRef] = []
        self.cache = _VideoCacheLRU(memory_cache_videos)
        self.include_gt = include_gt

        for video_id in read_video_ids(video_ids_path):
            raw_path = find_subtitle(cfg.paths.subtitles_raw, video_id)
            visual_path = cfg.paths.feature_cache / f"{video_id}.npz"
            text_path = text_cache_directory(cfg) / f"{video_id}.npz"
            if raw_path is None or not visual_path.exists() or not text_path.exists():
                print(f"[dataset] skip {video_id}: subtitle or cache missing")
                continue
            raw = read_subtitles(raw_path)
            verify_text_cache(text_path, cfg, video_id)
            gt_path = find_subtitle(cfg.paths.subtitles_gt, video_id) if self.include_gt else None
            corrected = read_subtitles(gt_path) if gt_path is not None else None
            paired_gt = pair_corrected_cues(raw, corrected)
            text = self.cache.text(text_path, video_id)
            if len(text.embeddings) != len(raw):
                raise ValueError(f"Text/cache count mismatch for {video_id}")
            if text.texts != [c.text for c in raw]:
                raise ValueError(f"Stale text feature cache for {video_id}; rescan text")
            visual = self.cache.visual(visual_path, video_id)
            if not (len(visual.starts) == len(visual.ends) and len(visual.starts) > 0):
                raise ValueError(f"Invalid visual segments for {video_id}")
            if not np.all(np.diff(visual.starts) >= 0):
                raise ValueError(f"Segment time order is invalid for {video_id}")
            for i, cue in enumerate(raw):
                self.refs.append(SampleRef(video_id, i, cue, paired_gt[i]))

        if not self.refs:
            raise RuntimeError("No usable subtitle alignment samples")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict:
        ref = self.refs[index]
        cfg = self.cfg
        visual = self.cache.visual(
            cfg.paths.feature_cache / f"{ref.video_id}.npz", ref.video_id,
        )
        text = self.cache.text(
            text_cache_directory(cfg) / f"{ref.video_id}.npz", ref.video_id,
        )
        candidate_idx = select_candidate_indices(
            visual.starts, visual.ends,
            rough_start=ref.rough.start, rough_end=ref.rough.end,
            radius=cfg.semantic.search_radius_seconds,
            min_candidates=cfg.semantic.min_candidate_segments,
            max_candidates=cfg.semantic.max_candidate_segments,
        )
        verified = ref.corrected
        return {
            "video_id": ref.video_id,
            "cue_index": ref.cue_index,
            "text": ref.rough.text,
            "rough_start": np.float32(ref.rough.start),
            "rough_end": np.float32(ref.rough.end),
            "gt_start": np.float32(verified.start if verified is not None else np.nan),
            "gt_end": np.float32(verified.end if verified is not None else np.nan),
            "text_embedding": torch.from_numpy(text.embeddings[ref.cue_index].astype(np.float32)),
            "pose": torch.from_numpy(visual.pose[candidate_idx].astype(np.float32)),
            "rgb": torch.from_numpy(visual.rgb[candidate_idx].astype(np.float32)),
            "segment_starts": torch.from_numpy(visual.starts[candidate_idx].astype(np.float32)),
            "segment_ends": torch.from_numpy(visual.ends[candidate_idx].astype(np.float32)),
            "global_segment_indices": torch.from_numpy(candidate_idx.astype(np.int64)),
        }


def collate_alignment_batch(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("Empty batch")
    B = len(batch)
    M = max(x["pose"].shape[0] for x in batch)
    pose = torch.zeros((B, M, *batch[0]["pose"].shape[1:]), dtype=torch.float32)
    rgb = torch.zeros((B, M, *batch[0]["rgb"].shape[1:]), dtype=torch.float32)
    starts, ends = torch.zeros((B, M)), torch.zeros((B, M))
    global_indices = torch.full((B, M), -1, dtype=torch.long)
    mask = torch.zeros((B, M), dtype=torch.bool)
    for i, sample in enumerate(batch):
        count = sample["pose"].shape[0]
        pose[i, :count] = sample["pose"]
        rgb[i, :count] = sample["rgb"]
        starts[i, :count] = sample["segment_starts"]
        ends[i, :count] = sample["segment_ends"]
        global_indices[i, :count] = sample["global_segment_indices"]
        mask[i, :count] = True
    return {
        "video_id": [x["video_id"] for x in batch],
        "cue_index": torch.tensor([x["cue_index"] for x in batch], dtype=torch.long),
        "text": [x["text"] for x in batch],
        "rough_start": torch.tensor([x["rough_start"] for x in batch]),
        "rough_end": torch.tensor([x["rough_end"] for x in batch]),
        "gt_start": torch.tensor([x["gt_start"] for x in batch]),
        "gt_end": torch.tensor([x["gt_end"] for x in batch]),
        "text_embedding": torch.stack([x["text_embedding"] for x in batch]),
        "pose": pose,
        "rgb": rgb,
        "candidate_mask": mask,
        "segment_starts": starts,
        "segment_ends": ends,
        "global_segment_indices": global_indices,
    }

