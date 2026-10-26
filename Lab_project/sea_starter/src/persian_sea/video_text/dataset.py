
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from persian_sea.io.subtitles import find_subtitle, read_subtitles, normalize_persian_text
from persian_sea.semantic.candidates import select_candidate_indices
from persian_sea.alignment_sources import AlignmentSourceSettings, load_alignment_cues
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


def _load_visual(path: Path) -> VisualArrays:
    with np.load(path, allow_pickle=False) as f:
        return VisualArrays(f['pose'].astype(np.float32), f['rgb'].astype(np.float32),
                            f['starts'].astype(np.float32), f['ends'].astype(np.float32))


def _load_text(path: Path) -> TextArrays:
    with np.load(path, allow_pickle=False) as f:
        return TextArrays(f['embeddings'].astype(np.float32), tuple(map(str, f['texts'].tolist())))


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
        ids = [s.strip() for s in path.read_text(encoding="utf-8").splitlines()
               if s.strip() and not s.lstrip().startswith("#")]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate source video IDs in split")
        self.refs: list[CueRef] = []
        self.cache = _LRUCaches()
        self.videos = ids
        for vid in ids:
            subs = find_subtitle(Path(cfg.subtitles_dir), vid)
            visual_path = Path(cfg.feature_cache_dir) / f"{vid}.npz"
            text_path = Path(cfg.text_cache_dir) / f"{vid}.npz"
            if subs is None or not visual_path.is_file() or not text_path.is_file():
                raise FileNotFoundError(
                    f"Missing subtitle/visual/text cache for {vid}. "
                    "Run pose, segmentation, RGB and text cache extraction first."
                )
            cache_metadata = None
            if cfg.require_text_cache_metadata:
                cache_metadata = validate_sidecar(text_path, expected={
                    "backend": cfg.text_backend, "model_name": cfg.text_model,
                    "instruction": cfg.text_instruction,
                }, strict=True)
            raw = read_subtitles(subs)
            aligned, alignment_file = load_alignment_cues(vid, raw, AlignmentSourceSettings(
                mode=cfg.pair_source, approved_dir=cfg.approved_subtitles_dir,
                sea_dir=cfg.sea_alignment_dir, semantic_dir=cfg.semantic_alignment_dir,
                max_shift_seconds=cfg.max_alignment_shift_seconds))
            with np.load(text_path, allow_pickle=False) as tc, np.load(visual_path, allow_pickle=False) as vc:
                if cache_metadata is not None:
                    expected_hash = identity_digest(cache_metadata["identity"])
                    if ("cache_identity_sha256" not in tc or
                        str(tc["cache_identity_sha256"].item()) != expected_hash):
                        raise ValueError(f"Visual-text cache NPZ/sidecar identity mismatch: {vid}")
                if (len(tc["embeddings"]) != len(raw)
                    or [str(t) for t in tc["texts"].tolist()] != [c.text for c in raw]):
                    raise ValueError(f"Stale text cache (texts/order): {vid}")
                starts, ends = vc["starts"], vc["ends"]
                if (not len(starts) or len(starts) != len(ends)
                    or not np.isfinite(starts).all() or not np.isfinite(ends).all()
                    or np.any(ends <= starts) or np.any(np.diff(starts) < 0)):
                    raise ValueError(f"Invalid ordered visual signs: {vid}")
                if (vc["pose"].shape[0] != len(starts) or vc["rgb"].shape[0] != len(starts)
                    or vc["pose"].ndim != 4 or vc["rgb"].ndim != 4
                    or vc["pose"].shape[-2:] != (75, 4)):
                    raise ValueError(f"Invalid visual feature shape for {vid}")
                if (not np.isfinite(vc["pose"]).all() or not np.isfinite(vc["rgb"]).all()
                    or not np.isfinite(tc["embeddings"]).all()):
                    raise ValueError(f"Nonfinite visual/text features for {vid}")
            for i,c in enumerate(raw):
                matching = aligned[i]
                if c.text and c.end > c.start:
                    self.refs.append(CueRef(vid,i,c.text,c.start,c.end,matching.start,matching.end))
        if not self.refs:
            raise ValueError("No usable video/text pairs")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self,index):
        r = self.refs[index]
        vc = self.cache.get(self.cache._vis, r.video_id,
                            Path(self.cfg.feature_cache_dir)/f"{r.video_id}.npz",_load_visual)
        tc = self.cache.get(self.cache._text, r.video_id,
                            Path(self.cfg.text_cache_dir)/f"{r.video_id}.npz",_load_text)
        idx = select_candidate_indices(
            vc.starts, vc.ends, r.matching_start, r.matching_end,
            radius=self.cfg.candidate_margin_seconds,
            min_candidates=self.cfg.min_candidate_segments,
            max_candidates=self.cfg.max_candidate_segments,
        )
        return {
            "video_id":r.video_id, "cue_index":r.cue_index,
            "text":r.text, "text_key":normalize_persian_text(r.text),
            "rough_start":r.weak_start, "rough_end":r.weak_end,
            "pose":torch.from_numpy(vc.pose[idx].astype(np.float32)),
            "rgb":torch.from_numpy(vc.rgb[idx].astype(np.float32)),
            "starts":torch.from_numpy(vc.starts[idx].astype(np.float32)),
            "ends":torch.from_numpy(vc.ends[idx].astype(np.float32)),
            "text_embedding":torch.from_numpy(tc.embeddings[r.cue_index].astype(np.float32)),
        }


def collate_video_text(batch):
    if not batch:
        raise ValueError("Empty batch")
    B, M = len(batch), max(item["pose"].shape[0] for item in batch)
    pose = torch.zeros(B,M,*batch[0]["pose"].shape[1:], dtype=torch.float32)
    rgb = torch.zeros(B,M,*batch[0]["rgb"].shape[1:], dtype=torch.float32)
    start = torch.zeros(B,M)
    end = torch.zeros(B,M)
    mask = torch.zeros(B,M,dtype=torch.bool)
    for i,item in enumerate(batch):
        k = item["pose"].shape[0]
        if item["pose"].shape[1:] != batch[0]["pose"].shape[1:] or item["rgb"].shape[1:] != batch[0]["rgb"].shape[1:]:
            raise ValueError("Inconsistent cache tensor shapes")
        pose[i,:k],rgb[i,:k] = item["pose"],item["rgb"]
        start[i,:k],end[i,:k] = item["starts"],item["ends"]
        mask[i,:k] = True
    return {"pose":pose,"rgb":rgb,"starts":start,"ends":end,"mask":mask,
            "text_embedding":torch.stack([i["text_embedding"] for i in batch]),
            "video_id":[i["video_id"] for i in batch],
            "cue_index":torch.tensor([i["cue_index"] for i in batch]),
            "text_key":[i["text_key"] for i in batch],
            "rough_start":torch.tensor([i["rough_start"] for i in batch]),
            "rough_end":torch.tensor([i["rough_end"] for i in batch])}

class VideoDiverseBatchSampler(Sampler[list[int]]):


    def __init__(self, refs: list[CueRef], batch_size: int, seed:int=42, drop_last:bool=True):
        self.refs,self.batch_size,self.seed,self.drop_last=refs,batch_size,seed,drop_last
        self.epoch=0
        self.groups:dict[str,list[int]]={}
        for i,r in enumerate(refs):
            self.groups.setdefault(r.video_id,[]).append(i)
        if len(self.groups)<2:
            raise ValueError("Contrastive learning requires at least two distinct source videos")
        if batch_size<2:
            raise ValueError("batch_size must be >=2")

    def set_epoch(self, epoch:int):
        self.epoch=epoch

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        pools={key:list(rng.permutation(indices)) for key,indices in self.groups.items()}
        while True:
            active=[k for k,v in pools.items() if v]
            if len(active)<2: break
            rng.shuffle(active)
            batch=[pools[v].pop() for v in active[:self.batch_size]]
            if len(batch)==self.batch_size or (not self.drop_last and len(batch)>=2):
                yield batch

    def __len__(self):
        return sum(map(len,self.groups.values()))//self.batch_size


def check_disjoint_splits(train_ids:str, val_ids:str, test_ids:str) -> dict:
    result={}
    sets={}
    for key,path in (("train",train_ids),("val",val_ids),("test",test_ids)):
        f=Path(path)
        if not f.is_file():
            raise FileNotFoundError(f)
        entries=[x.strip() for x in f.read_text(encoding="utf-8").splitlines()
                 if x.strip() and not x.lstrip().startswith("#")]
        if len(entries)!=len(set(entries)):
            raise ValueError(f"Duplicate video ID in {key}")
        sets[key]=set(entries)
        result[key]=len(entries)
    for a,b in (("train","val"),("train","test"),("val","test")):
        both=sets[a]&sets[b]
        if both:
            raise ValueError(f"Original video leakage {a}/{b}: {sorted(both)}")
    return result
