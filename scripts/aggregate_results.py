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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()
    source = Path(args.input)
    metric_file = source if source.is_file() else source / "metrics.json"
    metrics = json.loads(metric_file.read_text(encoding="utf-8")) if metric_file.exists() else {}
    rows = []
    for key, value in metrics.items():
        row = {"result_key": key}
        flatten("", value, row)
        rows.append(row)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}) if rows else ["result_key"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    md = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    md.extend("| " + " | ".join(str(row.get(c, "N/A")) for c in columns) + " |" for row in rows)
    (output / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    latex_rows = [" & ".join(columns) + r" \\"]
    latex_rows.extend(" & ".join(str(row.get(c, "N/A")) for c in columns) + r" \\" for row in rows)
    (output / "results.tex").write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    print(f"Wrote CSV, Markdown, and LaTeX tables to {output}")


if __name__ == "__main__":
    main()
