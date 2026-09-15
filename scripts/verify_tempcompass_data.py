#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TempCompass video coverage")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument(
        "--questions-root",
        type=Path,
        default=Path("external/TempCompass/questions"),
    )
    args = parser.parse_args()

    required = set()
    for path in args.questions_root.glob("*.json"):
        required.update(json.loads(path.read_text(encoding="utf-8")))
    found = {
        path.stem: path
        for path in args.video_root.rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES
    }
    missing = sorted(required - set(found))
    print(f"Required video IDs: {len(required)}")
    print(f"Resolved video IDs: {len(required) - len(missing)}")
    print(f"Missing video IDs:  {len(missing)}")
    for video_id in missing[:20]:
        print(f"  MISSING {video_id}.mp4")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
