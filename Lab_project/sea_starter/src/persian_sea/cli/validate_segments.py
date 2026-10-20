import argparse

from persian_sea.config import (
    load_config,
    read_video_ids,
    segmentation_subdir,
)
from persian_sea.segmentation.validator import (
    save_validation_reports,
    validate_eaf,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    prediction_dir = segmentation_subdir(cfg)
    reports = []

    for video_id in read_video_ids(cfg.paths.video_ids):
        eaf = prediction_dir / f"{video_id}.eaf"
        if not eaf.exists():
            print(f"[validate] missing: {eaf}")
            continue

        report = validate_eaf(eaf, cfg.elan.sign_tiers)
        reports.append(report)
        print(
            f"[validate] {video_id}: n={report.segment_count}, "
            f"median={report.median_duration}, warnings={report.warnings}"
        )

    output = cfg.paths.reports / "segmentation_validation.json"
    save_validation_reports(reports, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
