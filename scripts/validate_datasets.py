#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.run_benchmark import instantiate_loader, load_benchmark_configs


def validate_benchmark(name: str) -> dict:
    loader = instantiate_loader(name)
    by_task = defaultdict(lambda: {"annotation_samples": 0, "resolved_videos": 0, "missing_videos": 0})
    for sample in loader.iter_samples():
        row = by_task[sample.task]
        row["annotation_samples"] += 1
        resolved = bool(sample.metadata.get("video_resolved", Path(sample.video_path).is_file()))
        row["resolved_videos" if resolved else "missing_videos"] += 1
    return {
        "config": load_benchmark_configs()[name],
        "tasks": dict(sorted(by_task.items())),
        "totals": {
            key: sum(item[key] for item in by_task.values())
            for key in ("annotation_samples", "resolved_videos", "missing_videos")
        },
        "skipped_by_loader": 0,
        "note": "Counts include scoreable annotations with unresolved videos; unresolved items are not silently skipped.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/dataset_validation.json")
    args = parser.parse_args()
    report = {
        name: validate_benchmark(name)
        for name in ("vidhalluc", "videohallucer", "eventhallusion")
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
