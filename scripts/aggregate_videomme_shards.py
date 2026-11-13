#!/usr/bin/env python3
"""Validate and merge independently evaluated Video-MME shards."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.videomme.evaluator import evaluate_videomme
from src.data.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", choices=("short", "medium", "long"), required=True)
    parser.add_argument("--expected-per-task", type=int)
    args = parser.parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be positive")

    raw_root = PROJECT / "results" / "raw"
    summary_root = PROJECT / "results" / "summary"
    table_root = PROJECT / "results" / "tables"
    summary_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for method in args.methods:
        for task in args.tasks:
            records = []
            paths = []
            for shard in range(1, args.shards + 1):
                stem = (
                    f"{args.run_tag}_{method}_{task}_shard_{shard:02d}of{args.shards:02d}"
                    f"__qwen2.5-vl-7b__{method}__videomme__{task}.jsonl"
                )
                path = raw_root / stem
                if not path.is_file():
                    raise SystemExit(f"Missing shard output: {path}")
                paths.append(path)
                records.extend(read_jsonl(path))

            counts = Counter(record.sample_id for record in records)
            duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
            if duplicates:
                raise SystemExit(
                    f"Duplicate sample IDs for {method}/{task}: {duplicates[:10]}"
                )
            if any(
                record.method != method or record.task != task or record.benchmark != "videomme"
                for record in records
            ):
                raise SystemExit(f"Unexpected record metadata in {method}/{task} shards")
            if args.expected_per_task is not None and len(records) != args.expected_per_task:
                raise SystemExit(
                    f"{method}/{task}: got {len(records)} records, expected {args.expected_per_task}"
                )

            records.sort(key=lambda record: record.sample_id)
            merged_stem = f"{args.run_tag}_{method}_{task}"
            merged_path = raw_root / (
                f"{merged_stem}__qwen2.5-vl-7b__{method}__videomme__{task}.jsonl"
            )
            summary_path = summary_root / (
                f"{merged_stem}__qwen2.5-vl-7b__{method}__videomme__{task}.summary.json"
            )
            write_jsonl(merged_path, records)
            summary = evaluate_videomme(records)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            overall = summary["overall"]
            rows.append({
                "method": method,
                "task": task,
                "n": overall["n"],
                "correct": overall["correct"],
                "accuracy_percent": (
                    round(overall["accuracy"] * 100, 2)
                    if overall["accuracy"] is not None else ""
                ),
                "runtime_errors": overall["n_runtime_errors"],
                "invalid_answers": overall["n_invalid_answers"],
                "status": overall["status"],
                "raw_jsonl": str(merged_path),
                "summary_json": str(summary_path),
            })
            print(
                f"{method}/{task}: n={overall['n']} correct={overall['correct']} "
                f"accuracy={rows[-1]['accuracy_percent']}% status={overall['status']}"
            )

    table_path = table_root / f"{args.run_tag}_merged.results.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged table: {table_path}")


if __name__ == "__main__":
    main()
