#!/usr/bin/env python3
"""Create the fixed one-fifth MotionBench positive-feature tuning subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.motionbench import MotionBenchLoader
from src.experiments.subsets import (
    build_motionbench_tuning_subset_manifest,
    write_subset_manifest,
)
from src.utils.config import load_yaml


TASKS = (
    "action_order",
    "location_related_motion",
    "motion_recognition",
    "motion_related_objects",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument(
        "--output",
        default="manifests/motionbench_positive_tuning_seed42.json",
    )
    args = parser.parse_args()

    benchmark = load_yaml(PROJECT / "configs" / "benchmarks.yaml")["benchmarks"][
        "motionbench"
    ]
    loader = MotionBenchLoader(
        benchmark["meta_path"], benchmark["video_root"], list(TASKS)
    )
    manifest = build_motionbench_tuning_subset_manifest(
        samples=loader.iter_samples(),
        tasks=TASKS,
        seed=args.seed,
        fraction=args.fraction,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT / output
    write_subset_manifest(manifest, output)
    selections = manifest["selections"]
    print(json.dumps({
        "manifest": str(output),
        "seed": args.seed,
        "fraction": args.fraction,
        "records": {
            task: len(selections[f"motionbench/{task}"]["sample_ids"])
            for task in TASKS
        },
        "total": sum(len(value["sample_ids"]) for value in selections.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
