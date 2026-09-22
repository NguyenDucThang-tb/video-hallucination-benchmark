#!/usr/bin/env python3
"""Create a stable manifest containing all VideoHallucer TPH and SDH records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.videohallucer import VideoHallucerLoader
from src.experiments.subsets import SCHEMA_VERSION, write_subset_manifest
from src.utils.config import load_yaml


TASKS = ("tph", "sdh")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="manifests/videohallucer_positive_tuning_tph_sdh_full_seed42.json",
    )
    args = parser.parse_args()

    config = load_yaml(PROJECT / "configs/benchmarks.yaml")["benchmarks"]["videohallucer"]
    samples = list(
        VideoHallucerLoader(config["data_root"], list(TASKS)).iter_samples()
    )
    by_task = {task: [] for task in TASKS}
    for sample in samples:
        by_task[sample.task].append(sample.sample_id)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": 42,
        "benchmark": "videohallucer",
        "tasks": list(TASKS),
        "selection_policy": (
            "all annotated VideoHallucer records; both branches of every pair retained"
        ),
        "selections": {
            f"videohallucer/{task}": {
                "unit": "pair_branch",
                "requested_units": len(sample_ids),
                "selected_units": sample_ids,
                "sample_ids": sample_ids,
            }
            for task, sample_ids in by_task.items()
        },
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT / output
    write_subset_manifest(manifest, output)
    print(json.dumps({
        "manifest": str(output),
        "records_by_task": {task: len(ids) for task, ids in by_task.items()},
        "total_records": len(samples),
        "pairs_by_task": {
            task: len({sample_id.rsplit(":", 1)[0] for sample_id in sample_ids})
            for task, sample_ids in by_task.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
