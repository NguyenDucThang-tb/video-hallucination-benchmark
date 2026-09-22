#!/usr/bin/env python3
"""Print all EventHallusion Positive Feature grid points and task accuracies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def pct(value: str | None) -> str:
    return "N/A" if value in (None, "") else f"{float(value) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument(
        "--glob",
        default="eventhallusion_positive_grid_qwen2_5_vl_7b_p*.grid.csv",
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.tables_dir.glob(args.glob)):
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    print(
        f"{'Point':>5} {'alpha':>5} {'alpha_s':>7} {'beta':>5} "
        f"{'Entire':>10} {'Misleading':>12} {'Mix':>10} {'Mean':>10} "
        f"{'Records':>12} {'Status'}"
    )
    print("-" * 115)
    for row in rows:
        experiment = row.get("experiment", "")
        point = experiment.rsplit("_p", 1)[-1].split("__", 1)[0]
        print(
            f"{point:>5} {row.get('alpha', 'N/A'):>5} "
            f"{row.get('alpha_s', 'N/A'):>7} {row.get('beta', 'N/A'):>5} "
            f"{pct(row.get('eventhallusion_entire')):>10} "
            f"{pct(row.get('eventhallusion_misleading')):>12} "
            f"{pct(row.get('eventhallusion_mix')):>10} "
            f"{pct(row.get('mean_score')):>10} "
            f"{row.get('record_count', 0)}/{row.get('expected_records', 0):<6} "
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
            f"Entire={pct(best.get('eventhallusion_entire'))} "
            f"Misleading={pct(best.get('eventhallusion_misleading'))} "
            f"Mix={pct(best.get('eventhallusion_mix'))} "
            f"Mean={pct(best.get('mean_score'))}"
        )


if __name__ == "__main__":
    main()
