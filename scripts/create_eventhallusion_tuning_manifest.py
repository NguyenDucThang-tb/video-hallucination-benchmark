#!/usr/bin/env python3
"""Create a stable manifest containing every EventHallusion QA record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.eventhallusion import EventHallusionLoader
from src.experiments.subsets import SCHEMA_VERSION, write_subset_manifest
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="manifests/eventhallusion_positive_tuning_full_seed42.json",
    )
    args = parser.parse_args()

    config = load_yaml(PROJECT / "configs/benchmarks.yaml")["benchmarks"]["eventhallusion"]
    samples = list(
        EventHallusionLoader(
            config["questions_root"],
            config["video_root"],
            ["entire", "misleading", "mix"],
        ).iter_samples()
    )
    by_task: dict[str, list[str]] = {task: [] for task in ("entire", "misleading", "mix")}
    for sample in samples:
        by_task[sample.task].append(sample.sample_id)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": 42,
        "benchmark": "eventhallusion",
        "selection_policy": (
            "all annotated EventHallusion records; unresolved videos retained in denominator"
        ),
        "selections": {
            f"eventhallusion/{task}": {
                "unit": "question",
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
        "unresolved_records": sum(
            not sample.metadata.get("video_resolved", True) for sample in samples
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
