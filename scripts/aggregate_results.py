#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def _load_check_compatibility():
    source = PROJECT / "src" / "models" / "compatibility.py"
    spec = importlib.util.spec_from_file_location("benchmark_compatibility", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load compatibility matrix from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_compatibility


check_compatibility = _load_check_compatibility()


MODELS = ("llava-ov-7b", "qwen2.5-vl-7b", "llava-video-7b")
METHODS = ("base", "tcd", "dino_heal", "season")
DISPLAY_MODEL = {
    "llava-ov-7b": "LLaVA-OV-7B",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
    "llava-video-7b": "LLaVA-Video-7B",
}
DISPLAY_METHOD = {"base": "Base", "tcd": "TCD", "dino_heal": "DINO-HEAL", "season": "SEASON"}
COLUMNS = (
    "Models", "Training-free",
    "VidHalluc_BQA", "VidHalluc_MCQ", "VidHalluc_STH", "VidHalluc_TSH", "VidHalluc_AVG",
    "VideoHallucer_ORH", "VideoHallucer_TPH", "VideoHallucer_SDH", "VideoHallucer_EFH",
    "VideoHallucer_ENFH", "VideoHallucer_AVG", "EventHallusion_AVG",
)


def nested_get(mapping: dict, path: str):
    current = mapping
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def percent(value) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    return f"{100.0 * float(value):.2f}"


def build_final_rows(metrics: dict) -> list[dict[str, str]]:
    rows = []
    for model in MODELS:
        for method in METHODS:
            supported, _ = check_compatibility(model, method)
            grouped = {
                benchmark: metrics.get(f"{model}/{method}/{benchmark}", {})
                for benchmark in ("vidhalluc", "videohallucer", "eventhallusion")
            }
            if not supported:
                grouped = {benchmark: {} for benchmark in grouped}
            vh = grouped["vidhalluc"]
            vhr = grouped["videohallucer"]
            eh = grouped["eventhallusion"]
            rows.append({
                "Models": f"{DISPLAY_MODEL[model]} - {DISPLAY_METHOD[method]}",
                "Training-free": "Yes",
                "VidHalluc_BQA": percent(nested_get(vh, "bqa.accuracy")),
                "VidHalluc_MCQ": percent(nested_get(vh, "mcq.accuracy")),
                "VidHalluc_STH": percent(nested_get(vh, "sth.accuracy")),
                "VidHalluc_TSH": percent(nested_get(vh, "tsh.accuracy")),
                "VidHalluc_AVG": percent(nested_get(vh, "avg.accuracy")),
                "VideoHallucer_ORH": percent(nested_get(vhr, "orh.accuracy")),
                "VideoHallucer_TPH": percent(nested_get(vhr, "tph.accuracy")),
                "VideoHallucer_SDH": percent(nested_get(vhr, "sdh.accuracy")),
                "VideoHallucer_EFH": percent(nested_get(vhr, "efh.accuracy")),
                "VideoHallucer_ENFH": percent(nested_get(vhr, "enfh.accuracy")),
                "VideoHallucer_AVG": percent(nested_get(vhr, "avg.accuracy")),
                "EventHallusion_AVG": percent(nested_get(eh, "overall.accuracy")),
            })
    return rows


def write_markdown(rows: list[dict[str, str]], output: Path) -> None:
    lines = [
        "| Models | Training-free | VidHalluc |  |  |  |  | VideoHallucer |  |  |  |  |  | EventHallusion |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "|  |  | BQA | MCQ | STH | TSH | AVG | ORH | TPH | SDH | EFH | ENFH | AVG | AVG |",
    ]
    lines.extend("| " + " | ".join(row[column] for column in COLUMNS) + " |" for row in rows)
    lines.extend([
        "",
        "All values are percentages. N/A means not executed, incomplete, or unsupported.",
        "VidHalluc AVG is the strict macro-average of BQA/MCQ/official STH/TSH; VideoHallucer AVG is the macro-average of five strict pair accuracies; EventHallusion AVG is sample-weighted binary accuracy over all configured splits.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: list[dict[str, str]], output: Path) -> None:
    lines = [
        r"\begin{tabular}{llccccc|cccccc|c}",
        r"\toprule",
        r"\multirow{2}{*}{Models} & \multirow{2}{*}{Training-free} & \multicolumn{5}{c|}{VidHalluc} & \multicolumn{6}{c|}{VideoHallucer} & EventHallusion \\",
        r" & & BQA & MCQ & STH & TSH & AVG & ORH & TPH & SDH & EFH & ENFH & AVG & AVG \\",
        r"\midrule",
    ]
    for row in rows:
        values = [row[column].replace("_", r"\_") for column in COLUMNS]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()
    source = Path(args.input)
    metric_file = source if source.is_file() else source / "metrics.json"
    metrics = json.loads(metric_file.read_text(encoding="utf-8")) if metric_file.exists() else {}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = build_final_rows(metrics)
    with (output / "final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, output / "final_results.md")
    write_latex(rows, output / "final_results.tex")
    (output / "final_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote final_results.csv/.md/.tex/.json to {output}")


if __name__ == "__main__":
    main()
