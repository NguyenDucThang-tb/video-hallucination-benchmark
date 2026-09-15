#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark JSONL to the official TempCompass prediction schema"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nested = defaultdict(lambda: defaultdict(list))
    task = None
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("benchmark") != "tempcompass":
                raise ValueError(f"Not a TempCompass record: {record.get('sample_id')}")
            task = task or record["task"]
            if record["task"] != task:
                raise ValueError("Input JSONL contains more than one TempCompass task")
            metadata = record.get("metadata") or {}
            nested[metadata["video_id"]][metadata["temporal_aspect"]].append({
                "question": metadata["official_question"],
                "answer": metadata["official_answer"],
                "prediction": record.get("raw_output"),
                "question_index": metadata["question_index"],
            })

    output = {}
    for video_id, dimensions in nested.items():
        output[video_id] = {}
        for aspect, items in dimensions.items():
            items.sort(key=lambda item: item.pop("question_index"))
            output[video_id][aspect] = items

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {sum(len(items) for dims in output.values() for items in dims.values())} records")
    print(f"Task: {task}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
