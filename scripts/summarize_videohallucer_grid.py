#!/usr/bin/env python3
"""Print all VideoHallucer TPH/SDH Positive Feature grid results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def pct(value: str | None) -> str:
    return "N/A" if value in (None, "") else f"{float(value) * 100:.2f}%"


def raw_task_counts(raw_dir: Path, experiment: str) -> dict[str, tuple[int, int]]:
    latest = {}
    for path in raw_dir.glob(f"{experiment}__*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    row.get("sample_id"), row.get("model"), row.get("method"),
                    row.get("benchmark"), row.get("task"),
                )
                latest[key] = row
    counts = {}
    for task, expected in (("tph", 352), ("sdh", 400)):
        rows = [row for key, row in latest.items() if key[-1] == task]
        counts[task] = (len(rows), sum(bool(row.get("error")) for row in rows))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument(
        "--glob",
        default="videohallucer_positive_grid_qwen2_5_vl_7b_p*.grid.csv",
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.tables_dir.glob(args.glob)):
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    print(
        f"{'Point':>5} {'alpha':>5} {'alpha_s':>7} {'beta':>5} "
        f"{'TPH':>10} {'SDH':>10} {'Mean':>10} {'Records':>12} {'Status'}"
    )
    print("-" * 95)
    for row in rows:
        experiment = row.get("experiment", "")
        point = experiment.rsplit("_p", 1)[-1].split("__", 1)[0]
        counts = raw_task_counts(args.raw_dir, experiment)
        records = (
            f"TPH={counts['tph'][0]}/352 SDH={counts['sdh'][0]}/400"
            if any(counts[task][0] for task in counts)
            else f"{row.get('record_count', 0)}/{row.get('expected_records', 0)}"
        )
        print(
            f"{point:>5} {row.get('alpha', 'N/A'):>5} "
            f"{row.get('alpha_s', 'N/A'):>7} {row.get('beta', 'N/A'):>5} "
            f"{pct(row.get('videohallucer_tph')):>10} "
            f"{pct(row.get('videohallucer_sdh')):>10} "
            f"{pct(row.get('mean_score')):>10} "
            f"{records:<23} "
            f"{row.get('status', 'N/A')}"
        )

    complete = [
        row for row in rows
        if row.get("status") in {"complete", "complete_with_errors"}
        and row.get("mean_score") not in (None, "")
    ]
    print(f"\nPoints found: {len(rows)}/109")
    print(f"Complete:     {len(complete)}/109")
    if complete:
        best = max(complete, key=lambda row: float(row["mean_score"]))
        print("Best complete point:")
        print(
            f"  {best['experiment']} | "
            f"TPH={pct(best.get('videohallucer_tph'))} "
            f"SDH={pct(best.get('videohallucer_sdh'))} "
            f"Mean={pct(best.get('mean_score'))}"
        )


if __name__ == "__main__":
    main()
