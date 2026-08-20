#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.eventhallusion.evaluator import evaluate_binary
from src.benchmarks.vidhalluc.evaluator import evaluate_classification
from src.benchmarks.videohallucer.evaluator import pair_accuracy
from src.data.jsonl import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/metrics/metrics.json")
    parser.add_argument(
        "--experiment",
        help="Experiment-name prefix before '__'. Required when an input directory contains multiple experiments.",
    )
    args = parser.parse_args()
    source = Path(args.input)
    if source.is_file():
        files = [source]
    else:
        files = sorted(source.glob(f"{args.experiment}__*.jsonl" if args.experiment else "*.jsonl"))
        experiment_names = {path.name.split("__", 1)[0] for path in files if "__" in path.name}
        if args.experiment is None and len(experiment_names) > 1:
            names = ", ".join(sorted(experiment_names))
            raise SystemExit(
                f"Input contains multiple experiments ({names}). Pass --experiment to avoid mixing runs."
            )
    grouped = defaultdict(list)
    for path in files:
        for record in read_jsonl(path):
            grouped[(record.model, record.method, record.benchmark)].append(record)
    metrics = {}
    for (model, method, benchmark), records in grouped.items():
        key = f"{model}/{method}/{benchmark}"
        if benchmark == "vidhalluc":
            metrics[key] = evaluate_classification(records)
        elif benchmark == "videohallucer":
            metrics[key] = pair_accuracy(records)
        elif benchmark == "eventhallusion":
            metrics[key] = evaluate_binary(records)
        else:
            metrics[key] = {"status": "N/A", "reason": "evaluator adapter not configured"}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {destination} ({len(metrics)} groups)")


if __name__ == "__main__":
    main()
