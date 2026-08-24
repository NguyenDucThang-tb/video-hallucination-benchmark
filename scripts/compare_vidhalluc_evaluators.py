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

from src.evaluation.parsers import parse_vidhalluc_sth, parse_vidhalluc_tsh_official


def parse_args():
    parser = argparse.ArgumentParser(description="Compare vendored VidHalluc and local parsers per sample")
    parser.add_argument("--input", required=True, help="Prediction JSONL file or directory")
    parser.add_argument("--output-dir", default="results/audit")
    return parser.parse_args()


def load_upstream_tsh_parser():
    path = PROJECT / "external/VidHalluc/eval/evaluation/eval_tsh.py"
    spec = importlib.util.spec_from_file_location("vidhalluc_upstream_eval_tsh", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load upstream evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_answer_to_correct_answer


def upstream_sth_parse(text: str) -> tuple[str, str]:
    if ", Locations: " in text:
        scene_part, locations = text.split(", Locations: ", 1)
        return scene_part.split(": ", 1)[-1].split(",", 1)[0].strip(), locations.strip()
    return text.split(": ", 1)[-1].split(",", 1)[0].strip(), ""


def files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*__vidhalluc__*.jsonl"))


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    args = parse_args()
    upstream_tsh = load_upstream_tsh_parser()
    rows = []
    for path in files(Path(args.input)):
        for record in read_rows(path):
            task = record.get("task")
            raw_output = str(record.get("raw_output", ""))
            if task == "tsh":
                official_raw = upstream_tsh(raw_output)
                official = None if official_raw == "None" else official_raw
                local = parse_vidhalluc_tsh_official(raw_output).value
                gt = str(record.get("ground_truth", "")).strip().upper()
                official_correct = official is not None and official == gt
                local_correct = local is not None and local == gt
                difference = official != local or official_correct != local_correct
                rows.append({
                    "sample_id": record.get("sample_id"), "task": "tsh", "raw_output": raw_output,
                    "ground_truth": gt, "official_parsed_answer": official,
                    "local_parsed_answer": local, "official_correct": official_correct,
                    "local_correct": local_correct, "difference": difference,
                    "reason_for_difference": "parser_behavior" if difference else "",
                })
            elif task == "sth":
                official, official_locations = upstream_sth_parse(raw_output)
                local_result, local_locations = parse_vidhalluc_sth(raw_output)
                local = local_result.value
                gt = str(record.get("ground_truth", "")).strip().lower()
                official_correct = (official.lower() == "yes") == (gt == "yes")
                local_correct = local is not None and local == gt
                difference = official.lower() != (local or "") or official_locations != (local_locations or "")
                rows.append({
                    "sample_id": record.get("sample_id"), "task": "sth", "raw_output": raw_output,
                    "ground_truth": gt, "official_parsed_answer": official,
                    "local_parsed_answer": local, "official_locations": official_locations,
                    "local_locations": local_locations, "official_correct": official_correct,
                    "local_correct": local_correct,
                    "difference": difference,
                    "reason_for_difference": (
                        "upstream_treats_non-yes_as_negative_for_classification"
                        if difference and local is None
                        else ("format_or_location_parse" if difference else "")
                    ),
                })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id", "task", "ground_truth", "raw_output", "official_parsed_answer",
        "local_parsed_answer", "official_locations", "local_locations", "official_correct",
        "local_correct", "difference", "reason_for_difference",
    ]
    with (output_dir / "vidhalluc_evaluator_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "records": len(rows),
        "tsh_records": sum(row["task"] == "tsh" for row in rows),
        "sth_records": sum(row["task"] == "sth" for row in rows),
        "differences": sum(bool(row["difference"]) for row in rows),
        "upstream_source": "external/VidHalluc/eval/evaluation",
        "upstream_revision": "e753864f5c2500c38523f97992355e2352bf8732",
    }
    (output_dir / "vidhalluc_evaluator_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
