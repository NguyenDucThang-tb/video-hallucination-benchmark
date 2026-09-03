#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.experiments.positive_grid import (
    count_run_records,
    experiment_name,
    finalize_grid_rows,
    metric_scores,
    positive_feature_grid,
    safe_prefix,
    validate_run_diagnostics,
    write_grid_csv,
)
from src.experiments.subsets import load_subset_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Run the positive-feature ablation grid")
    parser.add_argument("--model", required=True, choices=("qwen2.5-vl-7b", "llava-ov-7b", "llava-video-7b"))
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Execute runs; otherwise only write the plan")
    parser.add_argument(
        "--smoke-ablations", action="store_true",
        help="Run one sample per task for one representative of each ablation",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--stop-index", type=int, default=None)
    return parser.parse_args()


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=PROJECT, env=os.environ.copy(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main():
    args = parse_args()
    manifest_path = Path(args.subset_manifest).expanduser().resolve()
    manifest = load_subset_manifest(manifest_path)
    seed = int(manifest["seed"])
    expected_records = sum(
        len(selection["sample_ids"]) for selection in manifest["selections"].values()
    )
    prefix = safe_prefix(args.prefix or f"{args.model}_positive_tuning_seed{seed}")
    if args.smoke_ablations:
        prefix = f"{prefix}_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    all_points = positive_feature_grid(args.include_baseline)
    if args.smoke_ablations:
        selected = []
        seen_ablations = set()
        for index, point in enumerate(all_points, 1):
            if point.ablation not in seen_ablations:
                selected.append((index, point))
                seen_ablations.add(point.ablation)
        args.execute = True
    else:
        stop = args.stop_index or len(all_points)
        selected = list(enumerate(all_points, 1))[args.start_index - 1:stop]
    if not selected:
        raise SystemExit("selected grid range is empty")

    config_dir = PROJECT / "manifests" / f"{prefix}_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, point in selected:
        name = experiment_name(prefix, point)
        config = {
            "name": name,
            "seed": seed,
            "models": [args.model],
            "methods": ["positive_feature"],
            "method_configs": {"positive_feature": {
                "alpha": point.alpha, "alpha_s": point.alpha_s, "beta": point.beta,
            }},
            "subset_manifest": str(manifest_path),
            "benchmarks": [
                {"name": "vidhalluc", "tasks": ["tsh", "mcq"]},
                {"name": "videohallucer", "tasks": ["tph"]},
                {"name": "eventhallusion", "tasks": ["entire", "misleading", "mix"]},
            ],
            "sampling": "configs/sampling.yaml",
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
                "num_beams": 1,
            },
            "output_dir": "results/raw",
            "manifest_dir": "manifests",
            "resume": args.resume,
        }
        config_path = config_dir / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        row = {
            "experiment": name, "ablation": point.ablation,
            "alpha": point.alpha, "alpha_s": point.alpha_s, "beta": point.beta,
            "status": "planned", "return_code": None, "error": "",
            "expected_records": expected_records,
        }
        rows.append(row)
        write_grid_csv(
            finalize_grid_rows(rows), PROJECT / "results" / "tables" / f"{prefix}.grid.csv"
        )
        print(f"[{index}/{len(all_points)}] {name}", flush=True)
        if args.execute:
            row["status"] = "running"
            write_grid_csv(
                finalize_grid_rows(rows), PROJECT / "results" / "tables" / f"{prefix}.grid.csv"
            )
            log_path = PROJECT / "logs" / f"{name}.log"
            return_code = _run([
                sys.executable, str(PROJECT / "scripts" / "run_benchmark.py"),
                "--config", str(config_path), "--allow-unvalidated", "--debug-errors",
                *(["--limit", "1"] if args.smoke_ablations else []),
            ], log_path)
            row["return_code"] = return_code
            metrics_path = PROJECT / "results" / "metrics" / f"{name}.metrics.json"
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                scores = metric_scores(metrics)
                row.update(scores)
                values = [scores[key] for key in ("tsh", "mcq", "tph", "eventhallusion")]
                row["mean_score"] = sum(values) / len(values) if all(v is not None for v in values) else None
                total, failed = count_run_records(PROJECT / "results" / "raw", name)
                diagnostic_errors = validate_run_diagnostics(
                    PROJECT / "results" / "raw", name, point
                )
                row["record_count"] = total
                row["failed_records"] = failed
                row["diagnostics_valid"] = not diagnostic_errors
                expected_for_run = expected_records
                if args.smoke_ablations:
                    event_tasks = {
                        sample_id.split(":", 1)[0]
                        for sample_id in manifest["selections"]["eventhallusion/*"]["sample_ids"]
                    }
                    expected_for_run = 3 + len(event_tasks)
                    row["expected_records"] = expected_for_run
                row["status"] = (
                    "complete"
                    if return_code == 0 and total == expected_for_run and failed == 0
                    and not diagnostic_errors
                    and (args.smoke_ablations or row["mean_score"] is not None)
                    else "failed"
                )
                if diagnostic_errors:
                    row["error"] = "; ".join(diagnostic_errors[:10])
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)
        write_grid_csv(finalize_grid_rows(rows), PROJECT / "results" / "tables" / f"{prefix}.grid.csv")

    output = write_grid_csv(
        finalize_grid_rows(rows), PROJECT / "results" / "tables" / f"{prefix}.grid.csv"
    )
    print(json.dumps({"grid_points": len(rows), "executed": args.execute, "summary_csv": str(output)}, indent=2))


if __name__ == "__main__":
    main()
