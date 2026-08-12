#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.models.compatibility import check_compatibility
from src.utils.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment1.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_plan(config: dict) -> list[dict]:
    plan = []
    for model in config["models"]:
        for method in config["methods"]:
            supported, note = check_compatibility(model, method)
            for benchmark in config["benchmarks"]:
                plan.append({
                    "model": model, "method": method, "benchmark": benchmark,
                    "status": "ready" if supported else "N/A", "note": note,
                })
    return plan


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT / config_path
    config = load_yaml(config_path)
    plan = build_plan(config)
    print(json.dumps({"config": str(config_path), "jobs": plan}, indent=2))
    if args.dry_run:
        return
    if args.smoke_test:
        from smoke_test import run_smoke
        run_smoke()
        return
    ready = [job for job in plan if job["status"] == "ready"]
    if not ready:
        raise SystemExit(
            "No GPU-validated model-method adapters are enabled. Run --dry-run and read "
            "docs/compatibility_matrix.md. No checkpoint was downloaded and no result was fabricated."
        )
    raise SystemExit("Adapter execution entry point is intentionally gated until a model passes GPU smoke validation.")


if __name__ == "__main__":
    main()
