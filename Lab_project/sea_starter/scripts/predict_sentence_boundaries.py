

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from persian_sea.translation.boundary import (
    build_boundary_features_for_inference,
    load_boundary_head,
    predict_windows_with_info,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Best trained sentence-boundary checkpoint")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, checkpoint = load_boundary_head(args.checkpoint, device)
    record = build_boundary_features_for_inference(
        args.video_id, cfg, checkpoint, device
    )
    windows, info = predict_windows_with_info(model, record, cfg, device)
    result = {
        "video_id": args.video_id,
        "source": "cached_rgb_pose_only_NO_SUBTITLES",
        "boundary_checkpoint": str(args.checkpoint),
        "method": "TDM_SCP_count_constrained" if cfg.use_scp else "legacy_boundary_peaks",
        "num_sign_segments": len(record["starts"]),
        "num_predicted_windows": len(windows),
        "count_prediction": asdict(info) if info is not None else None,
        "windows": windows,
    }
    dest = (Path(args.output) if args.output else
            Path(cfg.output_dir) / "predictions" / f"{args.video_id}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps({"output": str(dest),
                      "predicted_sentences": result["num_predicted_windows"],
                      "count_prediction": result["count_prediction"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
