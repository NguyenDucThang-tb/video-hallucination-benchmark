#!/usr/bin/env python3
"""Report progress for the large-Qwen benchmark matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.qwen25_vl_large import EXPECTED_RECORDS, METHODS, MODELS, iter_matrix, safe_slug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="qwen25_vl_large_v1")
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    args = parser.parse_args()

    print(f"{'Model':20} {'Method':18} {'Benchmark/Task':49} {'Records':>12} {'Errors':>7} Status")
    print("-" * 125)
    complete = incomplete = missing = 0
    for model, method, benchmark, task in iter_matrix():
        prefix = "_".join((args.run_tag, safe_slug(model), method, benchmark, safe_slug(task)))
        latest = {}
        for path in args.raw_dir.glob(f"{prefix}__*.jsonl"):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    latest[row.get("sample_id")] = row
        records = len(latest)
        errors = sum(bool(row.get("error")) for row in latest.values())
        expected = EXPECTED_RECORDS[benchmark][task]
        if records == 0:
            status = "MISSING"
            missing += 1
        elif records == expected and errors == 0:
            status = "COMPLETE"
            complete += 1
        elif records == expected:
            status = "COMPLETE_WITH_ERRORS"
            incomplete += 1
        else:
            status = "INCOMPLETE"
            incomplete += 1
        print(
            f"{model:20} {method:18} {benchmark + '/' + task:49} "
            f"{records:5d}/{expected:<6d} {errors:7d} {status}"
        )
    print(f"\nComplete={complete} Incomplete={incomplete} Missing={missing} Total={len(MODELS) * len(METHODS) * 22}")


if __name__ == "__main__":
    main()
