#!/usr/bin/env python3
"""Summarize TempCompass grid accuracy by task and fine-grained aspect."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.jsonl import read_jsonl


TASKS = ("multi-choice", "yes_no", "caption_matching")


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        aspect = record.metadata.get("fine_grained_aspect") or "unknown"
        groups[(record.task, aspect)].append(record)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument(
        "--glob",
        default="tempcompass_positive_grid_*_p*__*.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/tables/tempcompass_positive_grid_aspects.csv",
    )
    args = parser.parse_args()

    rows = []
    paths = sorted(Path(args.raw_dir).glob(args.glob))
    for path in paths:
        records = read_jsonl(path)
        if not records:
            continue
        groups = summarize(records)
        experiment = records[0].metadata.get("experiment", path.name.split("__", 1)[0])
        model = records[0].model
        for (task, aspect), items in sorted(groups.items()):
            if task not in TASKS:
                continue
            correct = sum(item.is_correct is True for item in items)
            errors = sum(item.error is not None for item in items)
            rows.append({
                "experiment": experiment,
                "model": model,
                "task": task,
                "fine_grained_aspect": aspect,
                "records": len(items),
                "correct": correct,
                "errors": errors,
                "parse_coverage": (len(items) - errors) / len(items),
                "accuracy": correct / len(items),
            })

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment", "model", "task", "fine_grained_aspect",
        "records", "correct", "errors", "parse_coverage", "accuracy",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {destination}")
    print(f"Rows:  {len(rows)}")
    for row in sorted(rows, key=lambda item: (-item["accuracy"], item["task"], item["fine_grained_aspect"]))[:20]:
        print(
            f"{row['model']:18} {row['task']:18} "
            f"{row['fine_grained_aspect']:24} "
            f"{row['accuracy'] * 100:6.2f}% "
            f"({row['correct']}/{row['records']})"
        )


if __name__ == "__main__":
    main()
