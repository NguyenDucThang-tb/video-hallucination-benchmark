#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MotionBench metadata and video coverage")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("external/MotionBench/data/video_info.meta.jsonl"),
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.meta_path.open(encoding="utf-8")
        if line.strip()
    ]
    required = {Path(str(row["video_path"])).name for row in rows}
    found = {
        path.name: path
        for path in args.video_root.rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES
    }
    split_counts = Counter()
    category_counts = Counter()
    for row in rows:
        for qa in row.get("qa", []):
            split = "test" if str(qa.get("answer", "")).strip().upper() == "NA" else "dev"
            split_counts[split] += 1
            category_counts[(split, str(row["question_type"]))] += 1

    missing = sorted(required - set(found))
    print(f"Metadata rows / QA: {len(rows)}")
    print(f"Unique videos:      {len(required)}")
    print(f"Dev / test QA:      {split_counts['dev']} / {split_counts['test']}")
    print(f"Resolved videos:    {len(required) - len(missing)}")
    print(f"Missing videos:     {len(missing)}")
    for split in ("dev", "test"):
        print(f"{split.upper()} categories:")
        for (item_split, category), count in sorted(category_counts.items()):
            if item_split == split:
                print(f"  {category}: {count}")
    for name in missing[:20]:
        print(f"  MISSING {name}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
