from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.prepare_server_splits import (
    available_videos, read_metadata, split_grouped,
    audit_group_leakage, write_splits,
)


def test_reproducible_without_video_leakage(tmp_path):
    groups = {f"rec-{i:02d}": [f"video-{i:02d}"] for i in range(20)}
    a = split_grouped(groups, seed=42)
    b = split_grouped(groups, seed=42)
    assert a == b
    assert len(a["train"]) >= 2 and len(a["val"]) >= 2 and len(a["test"]) >= 1
    assert len(set(sum(a.values(), []))) == 20
    write_splits(tmp_path, a, {"seed": 42}, overwrite=False)
    assert (tmp_path / "all_ids.txt").is_file()
    assert len((tmp_path / "all_ids.txt").read_text().splitlines()) == 20
    with pytest.raises(FileExistsError):
        write_splits(tmp_path, a, {}, overwrite=False)


def test_signer_independence_and_recording_rejection():
    metadata = {
        f"v{i}": {"signer_id": f"signer{i}", "source_recording_id": f"rec{i}"}
        for i in range(8)
    }
    groups = {f"signer{i}": [f"v{i}"] for i in range(8)}
    splits = split_grouped(groups, seed=17)
    results = audit_group_leakage(splits, metadata, strict_signer=True)
    assert not results["signer_id"]["cross_split_groups"]
    metadata["v7"]["source_recording_id"] = metadata["v0"]["source_recording_id"]
    # Two clips of the same original recording must not be split across train/test.
    for i in range(7):
        for j in range(i + 1, 8):
            if i == 0 and j == 7:
                a = {name for name, vids in splits.items() if f"v{i}" in vids}
                b = {name for name, vids in splits.items() if f"v{j}" in vids}
                if a != b:
                    with pytest.raises(ValueError, match="recording leakage"):
                        audit_group_leakage(splits, metadata, strict_signer=True)


def test_input_discovery_and_metadata_fail_closed(tmp_path):
    v = tmp_path / "videos"
    s = tmp_path / "subtitles"
    v.mkdir(); s.mkdir()
    for i in range(6):
        (v / f"v{i}.mp4").write_bytes(b"dummy")
        (s / f"v{i}.vtt").write_text("WEBVTT\n")
    ids = available_videos(v, s)
    assert len(ids) == 6
    meta = tmp_path / "meta.csv"
    with meta.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "signer_id", "source_recording_id"])
        writer.writeheader()
        for vid in ids[:5]:
            writer.writerow({"video_id": vid, "signer_id": "s1", "source_recording_id": "rec1"})
    with pytest.raises(ValueError, match="missing"):
        read_metadata(meta, ids)
