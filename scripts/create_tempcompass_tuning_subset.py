#!/usr/bin/env python3
"""Create one fixed, random one-third TempCompass tuning subset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from math import ceil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.tempcompass import TempCompassLoader
from src.experiments.subsets import write_subset_manifest
from src.utils.config import load_yaml


TASKS = ("multi-choice", "yes_no", "caption_matching")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1 / 3)
    parser.add_argument(
        "--output",
        default="manifests/tempcompass_positive_tuning_seed42.json",
    )
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise SystemExit("--fraction must be in (0, 1]")

    benchmark = load_yaml(PROJECT / "configs" / "benchmarks.yaml")["benchmarks"]["tempcompass"]
    loader = TempCompassLoader(
        benchmark["questions_root"],
        benchmark["video_root"],
        benchmark["meta_path"],
        list(TASKS),
    )
    grouped = {task: [] for task in TASKS}
    for sample in loader.iter_samples():
        if sample.metadata.get("video_resolved", True):
            grouped[sample.task].append(sample)

    rng = random.Random(args.seed)
    selections = {}
    for task in TASKS:
        samples = sorted(grouped[task], key=lambda sample: sample.sample_id)
        count = ceil(len(samples) * args.fraction)
        selected = sorted(rng.sample(samples, count), key=lambda sample: sample.sample_id)
        selections[f"tempcompass/{task}"] = {
            "unit": "instruction",
            "requested_fraction": args.fraction,
            "source_records": len(samples),
            "requested_records": count,
            "selected_units": [sample.sample_id for sample in selected],
            "sample_ids": [sample.sample_id for sample in selected],
        }

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "benchmark": "tempcompass",
        "tasks": list(TASKS),
        "selection_policy": "fixed random instruction subset; ceil(one-third) per task",
        "selections": selections,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT / output
    write_subset_manifest(manifest, output)
    print(json.dumps({
        "manifest": str(output),
        "seed": args.seed,
        "fraction": args.fraction,
        "records": {
            task: len(selections[f"tempcompass/{task}"]["sample_ids"])
            for task in TASKS
        },
        "total": sum(len(value["sample_ids"]) for value in selections.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
