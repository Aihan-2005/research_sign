
import argparse
from pathlib import Path
from persian_sea.config import load_config
from persian_sea.data.processing import run_steps


def main():
    ap = argparse.ArgumentParser(description="Run selected extraction stages (no temporal cropping)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--video-ids", help="Pilot ID file; default config's video_ids")
    ap.add_argument("--steps", nargs="+", required=True, help="pose segment visual text")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = run_steps(load_config(args.config), args.steps,
                      ids_file=Path(args.video_ids) if args.video_ids else None,
                      dry_run=args.dry_run)
    print("Validated" if args.dry_run else "Finished", ", ".join(names))


if __name__ == "__main__":
    main()
