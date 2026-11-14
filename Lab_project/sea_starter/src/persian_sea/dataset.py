from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from persian_sea.io.subtitles import find_subtitle, read_subtitles, normalize_persian_text
from persian_sea.semantic.candidates import select_candidate_indices
from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
from persian_sea.alignment.timing_policy import resolve_offset, shift_interval
from persian_sea.text.cache_identity import validate_sidecar, identity_digest
from .config import VideoTextConfig


@dataclass(frozen=True)
class CueRef:
    video_id: str
    cue_index: int
    text: str
    weak_start: float
    weak_end: float
    matching_start: float
    matching_end: float


@dataclass(frozen=True)
class VisualArrays:
    pose: np.ndarray
    rgb: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


@dataclass(frozen=True)
class TextArrays:
    embeddings: np.ndarray
    texts: tuple[str, ...]
    word_embeddings: np.ndarray | None = None
    word_cue_offsets: np.ndarray | None = None
    word_texts: tuple[str, ...] | None = None
    word_char_offsets: np.ndarray | None = None


def _load_visual(path: Path) -> VisualArrays:
    with np.load(path, allow_pickle=False) as f:
        return VisualArrays(
            f["pose"].astype(np.float32),
            f["rgb"].astype(np.float32),
            f["starts"].astype(np.float32),
            f["ends"].astype(np.float32),
        )


def _load_text(path: Path) -> TextArrays:
    with np.load(path, allow_pickle=False) as f:
        word_embeddings = (
            f["word_embeddings"].astype(np.float32)
            if "word_embeddings" in f.files else None
        )
        word_cue_offsets = (
            f["word_cue_offsets"].astype(np.int64)
            if "word_cue_offsets" in f.files else None
        )
        word_texts = (
            tuple(map(str, f["word_texts"].tolist()))
            if "word_texts" in f.files else None
        )
        word_char_offsets = (
            f["word_char_offsets"].astype(np.int32)
            if "word_char_offsets" in f.files else None
        )
        return TextArrays(
            embeddings=f["embeddings"].astype(np.float32),
            texts=tuple(map(str, f["texts"].tolist())),
            word_embeddings=word_embeddings,
            word_cue_offsets=word_cue_offsets,
            word_texts=word_texts,
            word_char_offsets=word_char_offsets,
        )


class _LRUCaches:
    def __init__(self, n: int = 2):
        self.n = n
        self._vis: OrderedDict[str, VisualArrays] = OrderedDict()
        self._text: OrderedDict[str, TextArrays] = OrderedDict()

    def get(self, cache: OrderedDict, key: str, path: Path, factory):
        if key not in cache:
            cache[key] = factory(path)
        cache.move_to_end(key)
        while len(cache) > self.n:
            cache.popitem(last=False)
        return cache[key]


class VideoTextDataset(Dataset):
    def __init__(self, cfg: VideoTextConfig, ids_file: str | Path):
        self.cfg = cfg.validate()
        path = Path(ids_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        ids = [
            s.strip()
            for s in path.read_text(encoding="utf-8").splitlines()
            if s.strip() and not s.lstrip().startswith("#")
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate source video IDs in split")
        self.refs: list[CueRef] = []
        self.cache = _LRUCaches()
        self.videos = ids
        self.needs_words = cfg.text_representation in {"word", "both"}

        for vid in ids:
            subs = find_subtitle(Path(cfg.subtitles_dir), vid)
            visual_path = Path(cfg.feature_cache_dir) / f"{vid}.npz"
            text_path = Path(cfg.text_cache_dir) / f"{vid}.npz"
            if subs is None or not visual_path.is_file() or not text_path.is_file():
                raise FileNotFoundError(
                    f"Missing subtitle/visual/text cache for {vid}. "
                    "Run pose, segmentation, RGB and text cache extraction first."
                )

            raw = read_subtitles(subs)
            raw_texts = [cue.text for cue in raw]

            cache_metadata = None
            if cfg.require_text_cache_metadata:
                expected = {
                    "backend": cfg.text_backend,
                    "model_name": cfg.text_model,
                    "instruction": cfg.text_instruction,
                    "model_revision": cfg.text_revision,
                    "max_length": cfg.text_max_length,
                }
                if self.needs_words:
                    expected.update({
                        "word_pooling": "mean_subword_l2",
                        "word_offset_schema": 1,
                    })
                cache_metadata = validate_sidecar(
                    text_path,
                    expected=expected,
                    strict=True,
                    texts=raw_texts,
                )

            aligned, alignment_file = load_alignment_cues(
                vid,
                raw,
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
                vid,
                cfg.pair_source,
                cfg.subtitle_to_sign_offset_seconds,
                cfg.subtitle_offsets_by_video,
            )

            with np.load(text_path, allow_pickle=False) as tc, np.load(
                visual_path, allow_pickle=False
            ) as vc:
                if cache_metadata is not None:
                    expected_hash = identity_digest(cache_metadata["identity"])
                    if (
                        "cache_identity_sha256" not in tc
                        or str(tc["cache_identity_sha256"].item()) != expected_hash
                    ):
                        raise ValueError(
                            f"Visual-text cache NPZ/sidecar identity mismatch: {vid}"
                        )

                cached_texts = [str(t) for t in tc["texts"].tolist()]
                if len(tc["embeddings"]) != len(raw) or cached_texts != raw_texts:
                    raise ValueError(f"Stale text cache (texts/order): {vid}")

                if self.needs_words:
                    required = {
                        "word_embeddings", "word_cue_offsets",
                        "word_texts", "word_char_offsets",
                    }
                    missing = required.difference(tc.files)
                    if missing:
                        raise ValueError(
                            f"Word-level cache is incomplete for {vid}; "
                            f"missing arrays: {sorted(missing)}. Re-run scan-text."
                        )
                    word_vectors = tc["word_embeddings"]
                    cue_offsets = np.asarray(tc["word_cue_offsets"], dtype=np.int64)
                    word_texts = tc["word_texts"]
                    char_offsets = np.asarray(tc["word_char_offsets"], dtype=np.int32)
                    if (
                        word_vectors.ndim != 2
                        or word_vectors.shape[1] != tc["embeddings"].shape[1]
                        or cue_offsets.shape != (len(raw) + 1,)
                        or cue_offsets[0] != 0
                        or np.any(np.diff(cue_offsets) < 0)
                        or cue_offsets[-1] != len(word_vectors)
                        or len(word_texts) != len(word_vectors)
                        or char_offsets.shape != (len(word_vectors), 2)
                    ):
                        raise ValueError(f"Invalid word-level cache shapes for {vid}")
                    if not np.isfinite(word_vectors).all():
                        raise ValueError(f"Nonfinite word embeddings for {vid}")
                    if len(char_offsets):
                        if np.any(char_offsets[:, 0] < 0) or np.any(
                            char_offsets[:, 1] <= char_offsets[:, 0]
                        ):
                            raise ValueError(f"Invalid word character offsets for {vid}")
                    for cue_index, cue in enumerate(raw):
                        a, b = cue_offsets[cue_index:cue_index + 2]
                        spans = char_offsets[a:b]
                        if len(spans) and np.any(spans[:, 1] > len(cue.text)):
                            raise ValueError(
                                f"Word character offset exceeds cue length: {vid}/{cue_index}"
                            )

                starts, ends = vc["starts"], vc["ends"]
                if (
                    not len(starts)
                    or len(starts) != len(ends)
                    or not np.isfinite(starts).all()
                    or not np.isfinite(ends).all()
                    or np.any(ends <= starts)
                    or np.any(np.diff(starts) < 0)
                ):
                    raise ValueError(f"Invalid ordered visual signs: {vid}")
                if (
                    vc["pose"].shape[0] != len(starts)
                    or vc["rgb"].shape[0] != len(starts)
                    or vc["pose"].ndim != 4
                    or vc["rgb"].ndim != 4
                    or vc["pose"].shape[-2:] != (75, 4)
                ):
                    raise ValueError(f"Invalid visual feature shape for {vid}")
                if (
                    not np.isfinite(vc["pose"]).all()
                    or not np.isfinite(vc["rgb"]).all()
                    or not np.isfinite(tc["embeddings"]).all()
                ):
                    raise ValueError(f"Nonfinite visual/text features for {vid}")

                word_counts = (
                    np.diff(cue_offsets) if self.needs_words
                    else np.zeros(len(raw), dtype=np.int64)
                )

            for i, cue in enumerate(raw):
                matching = aligned[i]
                if cue.text and cue.end > cue.start:
                    if self.needs_words and word_counts[i] == 0:
                        # A cue with no complete word after tokenizer truncation
                        # cannot contribute to the word-level objective.
                        continue
                    matching_start, matching_end = shift_interval(
                        matching.start, matching.end, offset
                    )
                    self.refs.append(
                        CueRef(
                            vid, i, cue.text, cue.start, cue.end,
                            matching_start, matching_end,
                        )
                    )

        if not self.refs:
            raise ValueError("No usable video/text pairs")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        ref = self.refs[index]
        vc = self.cache.get(
            self.cache._vis,
            ref.video_id,
            Path(self.cfg.feature_cache_dir) / f"{ref.video_id}.npz",
            _load_visual,
        )
        tc = self.cache.get(
            self.cache._text,
            ref.video_id,
            Path(self.cfg.text_cache_dir) / f"{ref.video_id}.npz",
            _load_text,
        )
        idx = select_candidate_indices(
            vc.starts,
            vc.ends,
            ref.matching_start,
            ref.matching_end,
            radius=self.cfg.candidate_margin_seconds,
            min_candidates=self.cfg.min_candidate_segments,
            max_candidates=self.cfg.max_candidate_segments,
        )
        item = {
            "video_id": ref.video_id,
            "cue_index": ref.cue_index,
            "text": ref.text,
            "text_key": normalize_persian_text(ref.text),
            "rough_start": ref.weak_start,
            "rough_end": ref.weak_end,
            "pose": torch.from_numpy(vc.pose[idx].astype(np.float32)),
            "rgb": torch.from_numpy(vc.rgb[idx].astype(np.float32)),
            "starts": torch.from_numpy(vc.starts[idx].astype(np.float32)),
            "ends": torch.from_numpy(vc.ends[idx].astype(np.float32)),
            "text_embedding": torch.from_numpy(
                tc.embeddings[ref.cue_index].astype(np.float32)
            ),
        }
        if self.needs_words:
            a, b = tc.word_cue_offsets[ref.cue_index:ref.cue_index + 2]
            item["word_embeddings"] = torch.from_numpy(
                tc.word_embeddings[a:b].astype(np.float32)
            )
            item["word_texts"] = list(tc.word_texts[a:b])
        return item


def collate_video_text(batch):
    if not batch:
        raise ValueError("Empty batch")
    batch_size = len(batch)
    max_segments = max(item["pose"].shape[0] for item in batch)
    pose = torch.zeros(
        batch_size, max_segments, *batch[0]["pose"].shape[1:], dtype=torch.float32
    )
    rgb = torch.zeros(
        batch_size, max_segments, *batch[0]["rgb"].shape[1:], dtype=torch.float32
    )
    start = torch.zeros(batch_size, max_segments)
    end = torch.zeros(batch_size, max_segments)
    mask = torch.zeros(batch_size, max_segments, dtype=torch.bool)

    for i, item in enumerate(batch):
        count = item["pose"].shape[0]
        if (
            item["pose"].shape[1:] != batch[0]["pose"].shape[1:]
            or item["rgb"].shape[1:] != batch[0]["rgb"].shape[1:]
        ):
            raise ValueError("Inconsistent cache tensor shapes")
        pose[i, :count], rgb[i, :count] = item["pose"], item["rgb"]
        start[i, :count], end[i, :count] = item["starts"], item["ends"]
        mask[i, :count] = True

    result = {
        "pose": pose,
        "rgb": rgb,
        "starts": start,
        "ends": end,
        "mask": mask,
        "text_embedding": torch.stack([item["text_embedding"] for item in batch]),
        "video_id": [item["video_id"] for item in batch],
        "cue_index": torch.tensor([item["cue_index"] for item in batch]),
        "text_key": [item["text_key"] for item in batch],
        "rough_start": torch.tensor([item["rough_start"] for item in batch]),
        "rough_end": torch.tensor([item["rough_end"] for item in batch]),
    }

    has_words = ["word_embeddings" in item for item in batch]
    if any(has_words) and not all(has_words):
        raise ValueError("A batch cannot mix sentence-only and word-level examples")
    if all(has_words):
        max_words = max(item["word_embeddings"].shape[0] for item in batch)
        if max_words < 1:
            raise ValueError("Word-level batch contains no valid words")
        word_dim = batch[0]["word_embeddings"].shape[-1]
        word_embeddings = torch.zeros(batch_size, max_words, word_dim)
        word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool)
        for i, item in enumerate(batch):
            words = item["word_embeddings"]
            if words.ndim != 2 or words.shape[-1] != word_dim or words.shape[0] < 1:
                raise ValueError("Invalid word embedding tensor in batch")
            count = words.shape[0]
            word_embeddings[i, :count] = words
            word_mask[i, :count] = True
        result["word_embeddings"] = word_embeddings
        result["word_mask"] = word_mask
        result["word_texts"] = [item["word_texts"] for item in batch]

    return result


class VideoDiverseBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        refs: list[CueRef],
        batch_size: int,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.refs = refs
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.groups: dict[str, list[int]] = {}
        for i, ref in enumerate(refs):
            self.groups.setdefault(ref.video_id, []).append(i)
        if len(self.groups) < 2:
            raise ValueError(
                "Contrastive learning requires at least two distinct source videos"
            )
        if batch_size < 2:
            raise ValueError("batch_size must be >=2")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {
            key: list(rng.permutation(indices))
            for key, indices in self.groups.items()
        }
        while True:
            active = [key for key, values in pools.items() if values]
            if len(active) < 2:
                break
            rng.shuffle(active)
            batch = [pools[key].pop() for key in active[:self.batch_size]]
            if len(batch) == self.batch_size or (
                not self.drop_last and len(batch) >= 2
            ):
                yield batch

    def __len__(self):
        return sum(map(len, self.groups.values())) // self.batch_size


def check_disjoint_splits(train_ids: str, val_ids: str, test_ids: str) -> dict:
    result = {}
    sets = {}
    for key, path in (
        ("train", train_ids), ("val", val_ids), ("test", test_ids)
    ):
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        entries = [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(entries) != len(set(entries)):
            raise ValueError(f"Duplicate video ID in {key}")
        sets[key] = set(entries)
        result[key] = len(entries)

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = sets[a] & sets[b]
        if both:
            raise ValueError(f"Original video leakage {a}/{b}: {sorted(both)}")
    return result
