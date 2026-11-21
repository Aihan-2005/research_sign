from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from scripts.audit_server_caches import audit_one, audit


def test_audit_one_valid_and_wrong_sample_count(tmp_path):
    vis, txt = tmp_path / "v.npz", tmp_path / "t.npz"
    np.savez_compressed(vis, pose=np.zeros((2, 8, 75, 4), dtype=np.float32),
                        rgb=np.zeros((2, 8, 4, 768), dtype=np.float32),
                        starts=np.array([0, 1], dtype=np.float32),
                        ends=np.array([0.9, 1.9], dtype=np.float32),
                        views=np.array(["signer", "left_hand", "right_hand", "face"]))
    np.savez_compressed(txt, embeddings=np.zeros((2, 1024), dtype=np.float32),
                        texts=np.array(["الف", "ب"]),
                        cache_identity_sha256=np.array("sha"))
    txt.with_suffix(".json").write_text(json.dumps({"identity": {"backend": "bge_m3"}}))
    expected = ["signer", "left_hand", "right_hand", "face"]
    good = audit_one("v", vis, txt, expected_k=8, expected_views=expected)
    assert not good["errors"]
    wrong = audit_one("v", vis, txt, expected_k=16, expected_views=expected)
    assert any("invalid visual NPZ" in message for message in wrong["errors"])
