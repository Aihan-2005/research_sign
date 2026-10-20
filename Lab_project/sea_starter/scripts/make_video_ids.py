from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="data/videos")
    parser.add_argument("--output", default="data/video_ids.txt")
    args = parser.parse_args()

    video_dir = Path(args.videos)
    ids = sorted(p.stem for p in video_dir.glob("*.mp4"))

    Path(args.output).write_text(
        "\n".join(ids) + ("\n" if ids else ""),
        encoding="utf-8",
    )

    print(f"Wrote {len(ids)} video ids to {args.output}")


if __name__ == "__main__":
    main()
