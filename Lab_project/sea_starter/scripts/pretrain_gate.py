
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from persian_sea.video_text.config import read_config
from persian_sea.config import load_config
from persian_sea.video_text.dataset import check_disjoint_splits
from persian_sea.video_text.split_audit import audit_signer_splits
from persian_sea.features.cache_audit import inspect_cache_pair
from persian_sea.translation.data import load_ids
from persian_sea.text.cache_identity import validate_sidecar


def gate(config: str, metadata: str = "", strict_signer: bool = False,
         app_config: str = "configs/baseline.yaml") -> dict:
    cfg = read_config(config)
    app = load_config(app_config)
    report = {"ok": False, "errors": [], "warnings": [], "splits": {}, "checks": {}}
    errors = report["errors"]
    try:
        counts = check_disjoint_splits(cfg.train_ids, cfg.val_ids, cfg.test_ids)
        report["splits"] = counts
        if counts["train"] < cfg.batch_size:
            errors.append(f"Need >= batch_size={cfg.batch_size} DISTINCT source videos in train")
        if counts["val"] < 2:
            errors.append("Existing VLP retrieval validation requires >=2 distinct source videos")
        if counts["test"] < 1:
            errors.append("Keep >=1 independent source video for final test")
        if metadata:
            report["signer_audit"] = audit_signer_splits(
                metadata, cfg.train_ids, cfg.val_ids, cfg.test_ids, strict_signer)
        elif strict_signer:
            errors.append("--strict-signer requires --metadata")
        else:
            report["warnings"].append("No signer metadata: cannot verify signer/session leakage")
    except (OSError, ValueError) as e:
        errors.append(f"Split audit: {e}")
        return report

    for split, file in (("train", cfg.train_ids), ("val", cfg.val_ids),
                        ("test", cfg.test_ids)):
        count = 0
        for vid in load_ids(file):
            row = inspect_cache_pair(
                vid, Path(cfg.feature_cache_dir), Path(cfg.text_cache_dir),
                Path(cfg.subtitles_dir),
                expected_rgb_model=app.scanner.rgb_model,
                expected_text_model=cfg.text_model,
                expected_samples=app.scanner.samples_per_segment,
                expected_views=app.scanner.resolved_views())
            serious = [x for x in row.get("findings", []) if x.get("severity") == "error"]
            for item in serious:
                errors.append(f"{split}/{vid}/{item.get('code')}: {item.get('detail', item.get('message'))}")
            if not serious:
                try:
                    validate_sidecar(Path(cfg.text_cache_dir) / (vid + ".npz"),
                                     expected={"backend": cfg.text_backend,
                                               "model_name": cfg.text_model,
                                               "instruction": cfg.text_instruction},
                                     strict=cfg.require_text_cache_metadata)
                    count += 1
                except (OSError, ValueError) as exc:
                    errors.append(f"{split}/{vid}: text cache sidecar mismatch: {exc}")
        report["checks"][split] = {"usable_videos": count}
    report["ok"] = not errors
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metadata", default="")
    ap.add_argument("--app-config", default="configs/baseline.yaml")
    ap.add_argument("--strict-signer", action="store_true")
    ap.add_argument("--report", default="outputs/reports/pretrain_gate.json")
    args = ap.parse_args()
    result = gate(args.config, args.metadata, args.strict_signer, args.app_config)
    p = Path(args.report)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Report:", p)
    if not result["ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
