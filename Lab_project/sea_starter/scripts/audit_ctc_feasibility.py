from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from persian_sea.translation.config import read_config
from persian_sea.translation.data import TranslationCueDataset, load_ids
from persian_sea.translation.language import load_tokenizer


def target_ids(tokenizer, text: str, cfg) -> list[int]:
    if cfg.translation_backend == "mbart50":
        encoded = tokenizer(
            text_target=[text], truncation=True,
            max_length=cfg.max_target_tokens,
        )
    else:
        encoded = tokenizer(
            [text], truncation=True, max_length=cfg.max_target_tokens,
        )
    return list(encoded["input_ids"][0])


def minimum_ctc_steps(ids: list[int]) -> int:

    repeats = sum(left == right for left, right in zip(ids, ids[1:]))
    return len(ids) + repeats


def audit_split(cfg, ids_path: str, tokenizer) -> dict:
    path = Path(ids_path)
    if not path.is_file():
        return {"status": "missing_split_file", "path": ids_path}
    dataset = TranslationCueDataset(cfg, path)
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        ids = target_ids(tokenizer, item["text"], cfg)
        source_steps = int(len(item["starts"]))
        required_steps = minimum_ctc_steps(ids)
        rows.append({
            "video_id": item["video_id"],
            "cue_index": int(item["cue_index"]),
            "source_steps": source_steps,
            "target_tokens": len(ids),
            "minimum_ctc_steps": required_steps,
            "feasible": source_steps >= required_steps,
        })
    if not rows:
        return {"status": "empty_split", "path": ids_path}
    source = np.asarray([row["source_steps"] for row in rows], dtype=np.float32)
    target = np.asarray([row["target_tokens"] for row in rows], dtype=np.float32)
    required = np.asarray([row["minimum_ctc_steps"] for row in rows], dtype=np.float32)
    feasible = sum(row["feasible"] for row in rows)
    return {
        "status": "ok",
        "path": ids_path,
        "samples": len(rows),
        "feasible_samples": feasible,
        "feasible_fraction": feasible / len(rows),
        "source_steps_median": float(np.median(source)),
        "target_tokens_median": float(np.median(target)),
        "minimum_ctc_steps_median": float(np.median(required)),
        "infeasible_examples": [row for row in rows if not row["feasible"]][:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether each subtitle target can fit the available visual-token "
            "length under the CTC minimum-length constraint. No model is trained."
        )
    )
    parser.add_argument("--config", default="configs/translation_mt5_first.yaml")
    parser.add_argument("--output", default="outputs/reports/ctc_feasibility.json")
    args = parser.parse_args()

    cfg = read_config(args.config)
    tokenizer = load_tokenizer(cfg)
    result = {
        "config": args.config,
        "constraint": "input_steps >= target_tokens + adjacent_repeated_target_tokens",
        "note": (
            "Counts use the configured tokenizer and candidate-selection path. "
            "This is a necessary CTC length check, not proof that CTC training will help."
        ),
        "splits": {
            "train": audit_split(cfg, cfg.train_ids, tokenizer),
            "validation": audit_split(cfg, cfg.val_ids, tokenizer),
            "test": audit_split(cfg, cfg.test_ids, tokenizer),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    for name, split in result["splits"].items():
        if split.get("status") == "ok":
            print(
                f"{name}: feasible={split['feasible_samples']}/{split['samples']} "
                f"({split['feasible_fraction']:.1%})"
            )
        else:
            print(f"{name}: {split['status']}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
