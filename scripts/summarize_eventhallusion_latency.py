#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXPECTED = {"entire": 114, "misleading": 102, "mix": 193}
MODELS = ("llava-ov-7b", "qwen2.5-vl-7b", "llava-video-7b")
METHODS = ("base", "tcd", "dino_heal", "season", "positive_feature")
METHOD_LABELS = {
    "base": "Base",
    "tcd": "+TCD",
    "dino_heal": "+DINO-HEAL",
    "season": "+SEASON",
    "positive_feature": "+Positive (Ours)",
}


@dataclass(frozen=True)
class TimedGroup:
    model: str
    method: str
    task: str
    records: int
    errors: int
    duplicates: int
    correct: int
    latencies: tuple[float, ...]


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def load_groups(raw_dir: Path, run_tag: str) -> dict[tuple[str, str, str], TimedGroup]:
    by_group: dict[tuple[str, str, str], list[tuple[int, dict]]] = defaultdict(list)
    sequence = 0
    for path in sorted(raw_dir.glob(f"{run_tag}_*__*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if row.get("benchmark") != "eventhallusion":
                    continue
                key = (row.get("model"), row.get("method"), row.get("task"))
                by_group[key].append((sequence, row))
                sequence += 1

    output = {}
    for key, entries in by_group.items():
        latest: dict[str, tuple[int, dict]] = {}
        for item in entries:
            latest[str(item[1].get("sample_id"))] = item
        rows = [row for _, row in sorted(latest.values(), key=lambda item: item[0])]
        latencies = tuple(
            float(row["runtime_seconds"])
            for row in rows
            if not row.get("error")
            and isinstance(row.get("runtime_seconds"), (int, float))
            and float(row["runtime_seconds"]) >= 0
        )
        output[key] = TimedGroup(
            model=key[0],
            method=key[1],
            task=key[2],
            records=len(rows),
            errors=sum(bool(row.get("error")) for row in rows),
            duplicates=len(entries) - len(rows),
            correct=sum(row.get("is_correct") is True for row in rows),
            latencies=latencies,
        )
    return output


def summarize_group(group: TimedGroup, warmup: int) -> dict[str, object]:
    steady = group.latencies[min(warmup, len(group.latencies)):]
    expected = EXPECTED[group.task]
    if group.records < expected:
        status = "INCOMPLETE"
    elif group.duplicates:
        status = "DUPLICATES"
    elif group.errors:
        status = "COMPLETE_WITH_ERRORS"
    else:
        status = "COMPLETE"
    return {
        "hardware": "NVIDIA H200 80GB",
        "model": group.model,
        "method": group.method,
        "task": group.task,
        "expected_records": expected,
        "records": group.records,
        "errors": group.errors,
        "duplicate_records": group.duplicates,
        "timed_records": len(group.latencies),
        "warmup_excluded": len(group.latencies) - len(steady),
        "correct": group.correct,
        "mean_all_seconds": statistics.fmean(group.latencies) if group.latencies else None,
        "mean_seconds": statistics.fmean(steady) if steady else None,
        "median_seconds": statistics.median(steady) if steady else None,
        "p95_seconds": percentile(steady, 0.95),
        "std_seconds": statistics.pstdev(steady) if len(steady) > 1 else 0.0 if steady else None,
        "accuracy": group.correct / group.records if group.records else None,
        "status": status,
    }


def format_seconds(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def build_markdown(rows: list[dict[str, object]], run_tag: str, warmup: int) -> str:
    lookup = {(row["model"], row["method"], row["task"]): row for row in rows}
    lines = [
        f"# EventHallusion latency on NVIDIA H200: `{run_tag}`",
        "",
        f"Batch size 1, 8 uniform frames, deterministic decoding, first {warmup} successful samples excluded per job.",
        "Latency is the benchmark runner's per-sample `generate_batch` wall time; model loading and video sampling are excluded.",
        "",
        "| Model | Method | Entire (s) | Misleading (s) | Mix (s) | Macro avg (s) | Weighted avg (s) | Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for method in METHODS:
            task_rows = [lookup.get((model, method, task)) for task in EXPECTED]
            means = [row["mean_seconds"] for row in task_rows if row and row["mean_seconds"] is not None]
            weighted_parts = [
                (row["mean_seconds"], row["timed_records"] - row["warmup_excluded"])
                for row in task_rows
                if row and row["mean_seconds"] is not None
            ]
            macro = statistics.fmean(means) if means else None
            weighted_n = sum(count for _, count in weighted_parts)
            weighted = (
                sum(value * count for value, count in weighted_parts) / weighted_n
                if weighted_n else None
            )
            total_records = sum(int(row["records"]) for row in task_rows if row)
            total_correct = sum(int(row["correct"]) for row in task_rows if row)
            accuracy = total_correct / total_records if total_records else None
            cells = [format_seconds(row["mean_seconds"] if row else None) for row in task_rows]
            accuracy_cell = "N/A" if accuracy is None else f"{accuracy * 100:.2f}%"
            lines.append(
                f"| {model} | {METHOD_LABELS[method]} | {cells[0]} | {cells[1]} | {cells[2]} | "
                f"{format_seconds(macro)} | {format_seconds(weighted)} | {accuracy_cell} |"
            )
    lines.extend([
        "",
        "## Coverage",
        "",
        "| Model | Method | Task | Records | Timed | Errors | Duplicates | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['model']} | {METHOD_LABELS.get(str(row['method']), str(row['method']))} | "
            f"{row['task']} | {row['records']}/{row['expected_records']} | {row['timed_records']} | "
            f"{row['errors']} | {row['duplicate_records']} | {row['status']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EventHallusion H200 latency runs")
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    groups = load_groups(args.raw_dir, args.run_tag)
    rows = [
        summarize_group(groups[key], args.warmup)
        for key in sorted(groups)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.run_tag}.latency.csv"
    markdown_path = args.output_dir / f"{args.run_tag}.latency.md"

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    markdown = build_markdown(rows, args.run_tag, args.warmup)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
