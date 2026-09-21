#!/usr/bin/env python3
"""Build or submit the Qwen2.5-VL 32B/72B benchmark matrix on Gadi."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.experiments.qwen25_vl_large import (
    BENCHMARK_TASKS,
    METHODS,
    MODELS,
    RESOURCE_PROFILES,
    iter_matrix,
    safe_slug,
)


PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=tuple(BENCHMARK_TASKS),
        default=list(BENCHMARK_TASKS),
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--run-tag", default="qwen25_vl_large_v1")
    parser.add_argument("--project", default="hn98")
    parser.add_argument("--queue", default="gpuhopper")
    parser.add_argument("--walltime", default="02:00:00")
    parser.add_argument("--smoke", action="store_true", help="Run 2 EventHallusion samples per model/method")
    parser.add_argument("--submit", action="store_true", help="Submit jobs; default is dry-run")
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=40,
        help="Submission guard; use 0 to disable after deliberately staging the run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = tuple(args.benchmarks)
    tasks = set(args.tasks) if args.tasks else None
    limit = None
    if args.smoke:
        benchmarks = ("eventhallusion",)
        tasks = {"entire"}
        limit = 2

    jobs = list(
        iter_matrix(
            models=tuple(args.models),
            methods=tuple(args.methods),
            benchmarks=benchmarks,
            tasks=tasks,
        )
    )
    if not jobs:
        raise SystemExit("No jobs matched the requested filters")
    if args.submit and args.max_jobs and len(jobs) > args.max_jobs:
        raise SystemExit(
            f"Refusing to submit {len(jobs)} jobs (guard={args.max_jobs}). "
            "Filter by model/benchmark or explicitly use --max-jobs 0."
        )

    pbs = PROJECT / "pbs/qwen25_vl_large_task_h200.pbs"
    logs = PROJECT / "logs"
    logs.mkdir(exist_ok=True)
    print(f"Mode: {'SUBMIT' if args.submit else 'DRY-RUN'} | jobs={len(jobs)}")

    for model, method, benchmark, task in jobs:
        profile = RESOURCE_PROFILES[model]
        prefix = "_".join(
            (args.run_tag, safe_slug(model), method, benchmark, safe_slug(task))
        )
        variables = {
            "MODEL": model,
            "METHOD": method,
            "BENCHMARK": benchmark,
            "TASK": task,
            "PREFIX": prefix,
        }
        if limit is not None:
            variables["LIMIT"] = str(limit)
        env_arg = ",".join(f"{key}={value}" for key, value in variables.items())
        command = [
            "qsub",
            "-P", args.project,
            "-q", args.queue,
            "-l", f"ncpus={profile.ncpus}",
            "-l", f"ngpus={profile.ngpus}",
            "-l", f"mem={profile.memory}",
            "-l", f"walltime={args.walltime}",
            "-v", env_arg,
            "-N", f"qvl_{safe_slug(model)}_{method}_{safe_slug(task)}"[:236],
            "-o", str(logs / f"{prefix}.out"),
            "-e", str(logs / f"{prefix}.err"),
            str(pbs),
        ]
        if args.submit:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"{model} / {method} / {benchmark}/{task} -> {result.stdout.strip()}")
        else:
            print(" ".join(command))


if __name__ == "__main__":
    main()
