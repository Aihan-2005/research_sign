
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import read_config


def main():
    parser = argparse.ArgumentParser(description="Gloss-free Persian translation baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "train", "infer", "eval"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "train":
            command.add_argument("--resume", default=None)
        if name == "eval":
            command.add_argument("--split", choices=("val", "test"), default="val")
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--output", default=None)
        if name == "infer":
            command.add_argument("--video-id", required=True)
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == "check":
        from .readiness import readiness_report, write_report
        report = readiness_report(cfg, require_all=True)
        output = Path(cfg.output_dir) / "readiness.json"
        write_report(report, output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("Wrote:", output)
        if report["errors"]:
            raise SystemExit(2)
    elif args.command == "train":
        from .train import run_training
        print("Best translation checkpoint:", run_training(cfg, args.resume))
    elif args.command == "infer":
        from .infer import generate_video
        print("Prediction JSON:", generate_video(cfg, args.video_id,
                                                  args.checkpoint, args.output))
    else:
        from .evaluate import evaluate_split
        print("Evaluation JSON:", evaluate_split(cfg, args.split,
                                                  args.checkpoint, args.output))


if __name__ == "__main__":
    main()
