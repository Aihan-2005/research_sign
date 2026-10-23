from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pilot_workflow.py"
    spec = importlib.util.spec_from_file_location("pilot_workflow", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_transport_archive_and_integrity(tmp_path, monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    files = {
        "video": "data/videos/v.mp4",
        "subtitle": "data/subtitles_raw/v.vtt",
        "pose": "data/poses/v.pose",
        "eaf": "outputs/segmentation/E4s-1_30_50/v.eaf",
        "text_npz": "outputs/text_features/bge_m3/v.npz",
        "text_json": "outputs/text_features/bge_m3/v.json",
        "visual_npz": "outputs/features/v.npz",
        "visual_json": "outputs/features/v.json",
    }
    
    paths = {k: Path(v) for k,v in files.items()}
    for path in paths.values():
        if path in (paths["visual_npz"], paths["visual_json"]):
            continue
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.as_posix().encode("utf-8"))
    monkeypatch.setattr(mod, "context", lambda cfg, video: (None, paths))
    output = tmp_path / "pilot.zip"
    mod.bundle(SimpleNamespace(id="v", config="ignored", output=str(output)))
    assert output.exists()
    with zipfile.ZipFile(output) as archive:
        meta = json.loads(archive.read("pilot_manifest.json"))
        assert meta["video_id"] == "v"
        assert len(meta["files"]) == 6
        assert archive.read("data/pilot_ids.txt") == b"v\n"
        for item in meta["files"]:
            assert mod.sha256_file(tmp_path / item["path"]) == item["sha256"]


def test_import_alignment_cost():
    from persian_sea.alignment.costs import alignment_cost
    from persian_sea.io.subtitles import SubtitleCue
    from persian_sea.types import SignSegment

    cue = SubtitleCue(1.0, 3.0, "سلام")
    signs = [SignSegment(1.0, 2.0), SignSegment(2.0, 3.0)]
    assert alignment_cost(cue, signs, 1.0, 1.0) == 0.0
