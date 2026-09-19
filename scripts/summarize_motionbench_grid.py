#!/usr/bin/env python3
"""Merge MotionBench positive-feature grid rows and report each model's best point."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


MODELS = {
    "qwen2_5_vl_7b": "qwen2.5-vl-7b",
    "llava_ov_7b": "llava-ov-7b",
    "llava_video_7b": "llava-video-7b",
}
TASK_COLUMNS = (
    "motionbench_action_order",
    "motionbench_location_related_motion",
    "motionbench_motion_recognition",
    "motionbench_motion_related_objects",
)


def _model(value: str) -> str:
    for slug, model in MODELS.items():
        if slug in value:
            return model
    return "unknown"


def _point(value: str) -> int | None:
    match = re.search(r"_p(\d+)(?:__|\.grid\.csv$)", value)
    return int(match.group(1)) if match else None


def _percentage(value: str | None) -> str:
    return "N/A" if value in (None, "") else f"{float(value) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument(
        "--glob", default="motionbench_positive_grid_*_p*.grid.csv"
    )
    parser.add_argument(
        "--output", default="results/tables/motionbench_positive_grid_all.csv"
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.tables_dir).glob(args.glob)):
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["model"] = _model(row.get("experiment", path.name))
                point = _point(row.get("experiment", "")) or _point(path.name)
                row["point"] = point
                rows.append(row)

    rows.sort(key=lambda row: (row["model"], row["point"] or 0))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "point", *next(iter(rows), {}).keys()]
    fields = list(dict.fromkeys(fields))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged rows: {len(rows)}/327")
    print(f"Output:      {destination}")
    print("\n===== BEST COMPLETE POINT PER MODEL =====")
    for model in MODELS.values():
        candidates = [
            row for row in rows
            if row["model"] == model
            and row.get("status") == "complete"
            and row.get("mean_score") not in (None, "")
        ]
        if not candidates:
            print(f"{model}: no complete point")
            continue
        best = max(candidates, key=lambda row: float(row["mean_score"]))
        task_scores = " | ".join(
            f"{column.removeprefix('motionbench_')}={_percentage(best.get(column))}"
            for column in TASK_COLUMNS
        )
        print(
            f"{model}: point {int(best['point']):03d} | "
            f"alpha={best['alpha']} alpha_s={best['alpha_s']} beta={best['beta']} | "
            f"{task_scores} | Mean={_percentage(best.get('mean_score'))}"
        )


if __name__ == "__main__":
    main()
