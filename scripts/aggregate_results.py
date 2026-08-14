#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten(prefix, value, output):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, output)
    else:
        output[prefix] = value


def nested_get(mapping: dict, path: str):
    current = mapping
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def build_paper_rows(metrics: dict) -> list[dict]:
    grouped = {}
    for result_key, value in metrics.items():
        model, method, benchmark = result_key.split("/", 2)
        grouped.setdefault((model, method), {})[benchmark] = value

    rows = []
    for (model, method), benchmarks in sorted(grouped.items()):
        vidhalluc = benchmarks.get("vidhalluc", {})
        videohallucer = benchmarks.get("videohallucer", {})
        eventhallusion = benchmarks.get("eventhallusion", {})
        rows.append({
            "Model": f"{model} + {method}",
            "VidHalluc_BQA": nested_get(vidhalluc, "bqa.accuracy"),
            "VidHalluc_MCQ": nested_get(vidhalluc, "mcq.accuracy") or nested_get(vidhalluc, "ach.accuracy"),
            "VidHalluc_STH": nested_get(vidhalluc, "sth.accuracy"),
            "VidHalluc_TSH": nested_get(vidhalluc, "tsh.accuracy"),
            "VidHalluc_AVG": nested_get(vidhalluc, "avg.accuracy"),
            "VideoHallucer_ORH": nested_get(videohallucer, "orh.accuracy"),
            "VideoHallucer_TPH": nested_get(videohallucer, "tph.accuracy"),
            "VideoHallucer_SDH": nested_get(videohallucer, "sdh.accuracy"),
            "VideoHallucer_EFH": nested_get(videohallucer, "efh.accuracy"),
            "VideoHallucer_ENFH": nested_get(videohallucer, "enfh.accuracy"),
            "VideoHallucer_AVG": nested_get(videohallucer, "avg.accuracy") or nested_get(videohallucer, "value"),
            "EventHallusion_entire": nested_get(eventhallusion, "entire.accuracy"),
            "EventHallusion_misleading": nested_get(eventhallusion, "misleading.accuracy"),
            "EventHallusion_AVG": nested_get(eventhallusion, "overall.accuracy"),
        })
    return rows


def stringify(value):
    if value is None:
        return ""
    return str(value)


def write_paper_markdown(rows: list[dict], output: Path) -> None:
    lines = [
        "| Model | VidHalluc |  |  |  |  | VideoHallucer |  |  |  |  |  | EventHallusion |  |  |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "|  | BQA | MCQ | STH | TSH | AVG | ORH | TPH | SDH | EFH | ENFH | AVG | entire | misleading | AVG |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join([
                stringify(row["Model"]),
                stringify(row["VidHalluc_BQA"]),
                stringify(row["VidHalluc_MCQ"]),
                stringify(row["VidHalluc_STH"]),
                stringify(row["VidHalluc_TSH"]),
                stringify(row["VidHalluc_AVG"]),
                stringify(row["VideoHallucer_ORH"]),
                stringify(row["VideoHallucer_TPH"]),
                stringify(row["VideoHallucer_SDH"]),
                stringify(row["VideoHallucer_EFH"]),
                stringify(row["VideoHallucer_ENFH"]),
                stringify(row["VideoHallucer_AVG"]),
                stringify(row["EventHallusion_entire"]),
                stringify(row["EventHallusion_misleading"]),
                stringify(row["EventHallusion_AVG"]),
            ]) + " |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_latex(rows: list[dict], output: Path) -> None:
    columns = [
        "Model", "VidHalluc_BQA", "VidHalluc_MCQ", "VidHalluc_STH", "VidHalluc_TSH", "VidHalluc_AVG",
        "VideoHallucer_ORH", "VideoHallucer_TPH", "VideoHallucer_SDH", "VideoHallucer_EFH",
        "VideoHallucer_ENFH", "VideoHallucer_AVG", "EventHallusion_entire",
        "EventHallusion_misleading", "EventHallusion_AVG",
    ]
    lines = [" & ".join(columns) + r" \\"]
    for row in rows:
        lines.append(" & ".join(stringify(row[column]) for column in columns) + r" \\")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    source = Path(args.input)
    metric_file = source if source.is_file() else source / "metrics.json"
    metrics = json.loads(metric_file.read_text(encoding="utf-8")) if metric_file.exists() else {}

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    detailed_rows = []
    for key, value in metrics.items():
        row = {"result_key": key}
        flatten("", value, row)
        detailed_rows.append(row)
    detailed_columns = sorted({key for row in detailed_rows for key in row}) if detailed_rows else ["result_key"]
    with (output / "detailed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detailed_columns)
        writer.writeheader()
        writer.writerows(detailed_rows)

    paper_rows = build_paper_rows(metrics)
    paper_columns = [
        "Model",
        "VidHalluc_BQA", "VidHalluc_MCQ", "VidHalluc_STH", "VidHalluc_TSH", "VidHalluc_AVG",
        "VideoHallucer_ORH", "VideoHallucer_TPH", "VideoHallucer_SDH", "VideoHallucer_EFH",
        "VideoHallucer_ENFH", "VideoHallucer_AVG",
        "EventHallusion_entire", "EventHallusion_misleading", "EventHallusion_AVG",
    ]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paper_columns)
        writer.writeheader()
        writer.writerows(paper_rows)

    write_paper_markdown(paper_rows, output / "results.md")
    write_paper_latex(paper_rows, output / "results.tex")
    print(f"Wrote paper-style CSV/Markdown/LaTeX tables to {output}")


if __name__ == "__main__":
    main()
