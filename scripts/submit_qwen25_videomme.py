#!/usr/bin/env python3
"""Dry-run or submit the Qwen2.5-VL-7B Video-MME method x duration matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
METHODS = ("base", "tcd", "dino_heal", "season", "positive_feature")
TASKS = ("short", "medium", "long")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--env-archive", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--run-tag", default="qwen25_videomme_8frame_v1")
    parser.add_argument("--project", default="hn98")
    parser.add_argument("--queue", default="gpuhopper")
    parser.add_argument("--walltime", default="04:00:00")
    parser.add_argument("--shards", type=int, default=1, help="Disjoint sample shards per method and duration")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--smoke", action="store_true", help="Submit at most two examples for each selected method")
    parser.add_argument("--submit", action="store_true", help="Submit jobs; without this flag only print qsub commands")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    if not data_root.is_dir():
        raise SystemExit(f"Video-MME dataset root does not exist: {data_root}")
    methods = args.methods
    tasks = args.tasks[:1] if args.smoke else args.tasks
    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    if args.smoke and args.shards != 1:
        raise SystemExit("--smoke and --shards cannot be combined")
    pbs = PROJECT / "pbs/qwen25_videomme_task_h200.pbs"
    logs = PROJECT / "logs"
    logs.mkdir(exist_ok=True)
    print(f"Mode={'SUBMIT' if args.submit else 'DRY-RUN'} | jobs={len(methods) * len(tasks) * args.shards}")

    for method in methods:
        for task in tasks:
            for shard_index in range(args.shards):
                prefix = f"{args.run_tag}_{method}_{task}"
                if args.shards > 1:
                    prefix += f"_shard_{shard_index + 1:02d}of{args.shards:02d}"
                variables = {
                    "METHOD": method,
                    "TASK": task,
                    "PREFIX": prefix,
                    "VIDEOMME_DATA_ROOT": str(data_root),
                    "ENV_ARCHIVE": str(Path(args.env_archive).expanduser()),
                    "HF_HOME": str(Path(args.hf_home).expanduser()),
                    "HF_HUB_CACHE": str(Path(args.hf_home).expanduser()),
                }
                if args.shards > 1:
                    variables["SHARD_INDEX"] = str(shard_index)
                    variables["SHARD_COUNT"] = str(args.shards)
                if args.smoke:
                    variables["LIMIT"] = "2"
                label = f"{method}/{task}"
                if args.shards > 1:
                    label += f" shard {shard_index + 1}/{args.shards}"
                command = [
                    "qsub", "-P", args.project, "-q", args.queue,
                    "-l", "ncpus=12", "-l", "ngpus=1", "-l", "mem=96GB",
                    "-l", "jobfs=40GB", "-l", f"walltime={args.walltime}",
                    "-l", "storage=scratch/jp09+gdata/hn98",
                    "-v", ",".join(f"{key}={value}" for key, value in variables.items()),
                    "-N", f"q25vm_{method}_{task}_s{shard_index + 1}"[:236],
                    "-o", str(logs / f"{prefix}.out"),
                    "-e", str(logs / f"{prefix}.err"), str(pbs),
                ]
                if args.submit:
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"{label} -> {result.stdout.strip()}")
                else:
                    print(" ".join(command))


if __name__ == "__main__":
    main()
