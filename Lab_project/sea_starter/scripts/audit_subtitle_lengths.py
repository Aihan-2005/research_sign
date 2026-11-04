

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persian_sea.io.subtitles import find_subtitle, read_subtitles


def token_count(tokenizer, text: str, *, target: bool = False) -> int:

    kwargs = {"truncation": False, "padding": False, "add_special_tokens": True}
    try:
        encoded = tokenizer(text_target=text, **kwargs) if target else tokenizer(text, **kwargs)
    except TypeError:
        encoded = tokenizer(text, **kwargs)
    return len(encoded["input_ids"])


def audit_cues(video_id, cues, text_tok, target_tok, text_limit: int, target_limit: int):
    rows = []
    for i, cue in enumerate(cues):
        a = token_count(text_tok, cue.text)
        b = token_count(target_tok, cue.text, target=True)
        rows.append({"video_id": video_id, "cue_index": i,
                     "duration_s": round(cue.end-cue.start, 4),
                     "characters": len(cue.text),
                     "bge_tokens_full": a, "generation_tokens_full": b,
                     "bge_would_truncate": a > text_limit,
                     "target_would_truncate": b > target_limit,
                     "text_preview": cue.text[:120]})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", default="data/train_ids.txt")
    p.add_argument("--subtitle-dir", default="data/subtitles_raw")
    p.add_argument("--text-model", default="BAAI/bge-m3")
    p.add_argument("--generator-model", default="google/mt5-small")
    p.add_argument("--text-limit", type=int, default=256)
    p.add_argument("--target-limit", type=int, default=96)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--output", default="outputs/reports/subtitle_lengths.json")
    args = p.parse_args()
    if min(args.text_limit, args.target_limit) < 1:
        raise ValueError("Token limits must be positive")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install project transformers dependencies first") from exc
    text_tok = AutoTokenizer.from_pretrained(args.text_model,
                                              local_files_only=args.local_files_only)
    target_tok = AutoTokenizer.from_pretrained(args.generator_model,
                                                local_files_only=args.local_files_only)
    ids = [s.strip() for s in Path(args.ids).read_text(encoding="utf-8").splitlines()
           if s.strip() and not s.lstrip().startswith("#")]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video IDs")
    rows = []
    for vid in ids:
        path = find_subtitle(Path(args.subtitle_dir), vid)
        if path is None:
            raise FileNotFoundError(f"Subtitle missing: {vid}")
        rows.extend(audit_cues(vid, read_subtitles(path), text_tok, target_tok,
                               args.text_limit, args.target_limit))
    bad_text = [r for r in rows if r["bge_would_truncate"]]
    bad_target = [r for r in rows if r["target_would_truncate"]]
    report = {"videos": len(ids), "cues": len(rows),
              "bge_truncated_count": len(bad_text),
              "target_truncated_count": len(bad_target),
              "text_model": args.text_model, "generator_model": args.generator_model,
              "text_limit": args.text_limit, "target_limit": args.target_limit,
              "rows": rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Tokens: {len(rows)} cues, BGE truncation={len(bad_text)}, "
          f"target truncation={len(bad_target)}; report={out}")
    if bad_text or bad_target:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
