#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.motionbench.parsers import polish_answer
from src.data.jsonl import read_jsonl
from src.evaluation.records import latest_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MotionBench test predictions")
    parser.add_argument("raw_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    records, duplicates = latest_records(read_jsonl(args.raw_jsonl))
    predictions = {}
    invalid = []
    for record in records:
        if record.metadata.get("evaluation_split") != "test":
            continue
        uid = str(record.metadata.get("motionbench_uid", record.sample_id))
        answer = record.normalized_output or polish_answer(record.raw_output)
        if answer not in {"A", "B", "C", "D"}:
            invalid.append(uid)
            continue
        predictions[uid] = answer

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Exported predictions: {len(predictions)}")
    print(f"Invalid/missing:       {len(invalid)}")
    print(f"Duplicates ignored:   {duplicates}")
    print(f"Output:                {args.output_json}")
    if invalid:
        print("First invalid UIDs:", ", ".join(invalid[:20]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
