
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")
ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


@dataclass(frozen=True)
class Recording:
    video_id: str
    source_recording_id: str
    signer_id: str
    split: str


def read_manifest(path: str | Path) -> list[Recording]:

    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        required = {"video_id", "source_recording_id", "signer_id", "split"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(f"Missing manifest columns: {sorted(required - set(rows.fieldnames or []))}")
        records = []
        for n, row in enumerate(rows, start=2):
            item = Recording(**{field: (row[field] or "").strip() for field in required})
            if not ID_RE.fullmatch(item.video_id) or item.video_id in {".", ".."}:
                raise ValueError(f"Invalid video_id at row {n}: {item.video_id!r}")
            if not item.source_recording_id:
                raise ValueError(f"Missing source_recording_id at row {n}; cannot prevent leakage")
            if item.split not in SPLITS:
                raise ValueError(f"Invalid split at row {n}: {item.split!r}")
            records.append(item)
    if not records:
        raise ValueError("Empty manifest")
    if len({r.video_id for r in records}) != len(records):
        raise ValueError("Duplicate video_id in manifest")
    return records


def audit_manifest(records: list[Recording], strict_signer: bool = False) -> dict:

 
    recordings = defaultdict(set)
    signers = defaultdict(set)
    counts = {x: 0 for x in SPLITS}
    unknown_signers = []
    for row in records:
        recordings[row.source_recording_id].add(row.split)
        counts[row.split] += 1
        if row.signer_id:
            signers[row.signer_id].add(row.split)
        else:
            unknown_signers.append(row.video_id)
    leaked = {key: sorted(vals) for key, vals in recordings.items() if len(vals) > 1}
    if leaked:
        raise ValueError(f"Source recording leakage between splits: {leaked}")
    shared_signers = {key: sorted(vals) for key, vals in signers.items() if len(vals) > 1}
    if strict_signer and (shared_signers or unknown_signers):
        raise ValueError(f"Signer-disjoint split failed: {shared_signers}, unknown={unknown_signers}")
    return {
        "split_videos": counts,
        "total_videos": len(records),
        "unique_source_recordings": len(recordings),
        "unique_signers_known": len(signers),
        "unknown_signers": sorted(unknown_signers),
        "signers_across_splits": shared_signers,
    }


def write_id_files(records: list[Recording], directory: str | Path, *, overwrite: bool = False) -> dict:

    directory = Path(directory)
    items = {f"{key}_ids.txt": sorted(r.video_id for r in records if r.split == key)
             for key in SPLITS}
    items["video_ids.txt"] = sorted(r.video_id for r in records)
    targets = {name: directory / name for name in items}
    exists = [str(p) for p in targets.values() if p.exists()]
    if exists and not overwrite:
        raise FileExistsError(f"Existing ID files: {exists}. Use --overwrite deliberately.")
    directory.mkdir(parents=True, exist_ok=True)
    for name, ids in items.items():

        target = targets[name]
        temp = target.with_name(target.name + ".tmp")
        temp.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        temp.replace(target)
    return {name: str(path) for name, path in targets.items()}
