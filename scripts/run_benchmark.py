#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.eventhallusion import EventHallusionLoader, evaluate_binary
from src.benchmarks.videohallucer import VideoHallucerLoader, pair_accuracy
from src.benchmarks.vidhalluc import VidHallucLoader, evaluate_classification
from src.data.jsonl import append_jsonl, read_jsonl, valid_resume_keys
from src.data.sampler import sample_video
from src.data.schema import PredictionRecord
from src.evaluation.normalize import normalize_prediction
from src.methods.base import BaseMethod
from src.methods.dino_heal.dino_heal_method import DINOHealMethod
from src.methods.season.season_method import SeasonMethod
from src.methods.tcd import TCDMethod
from src.models import GenerationConfig, LlavaOVAdapter, Qwen25VLAdapter
from src.models.compatibility import check_compatibility
from src.utils.config import load_yaml


DEBUG_ERRORS = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment1.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug-errors", action="store_true")
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Run adapters marked pending validation; intended for smoke tests only",
    )
    return parser.parse_args()


def build_plan(config: dict, allow_unvalidated: bool = False) -> list[dict]:
    plan = []
    benchmark_configs = load_benchmark_configs()
    for benchmark_entry in config["benchmarks"]:
        if isinstance(benchmark_entry, str):
            benchmark = benchmark_entry
            benchmark_tasks = benchmark_configs[benchmark].get("tasks") or [None]
        else:
            benchmark = benchmark_entry["name"]
            benchmark_tasks = benchmark_entry.get("tasks") or benchmark_configs[benchmark].get("tasks") or [None]
        for task in benchmark_tasks:
            for model in config["models"]:
                for method in config["methods"]:
                    supported, note = check_compatibility(model, method)
                    plan.append({
                        "model": model,
                        "method": method,
                        "benchmark": benchmark,
                        "task": task,
                        "status": "ready" if supported or allow_unvalidated else "N/A",
                        "note": note,
                        "validation_override": bool(allow_unvalidated and not supported),
                    })
    return plan


def load_model_configs() -> dict:
    return load_yaml(PROJECT / "configs/models.yaml")["models"]


def load_method_configs() -> dict:
    return load_yaml(PROJECT / "configs/methods.yaml")["methods"]


def resolve_method_config(name: str) -> dict:
    config = dict(load_method_configs()[name])
    override = os.environ.get(f"{name.upper()}_BATCH_SIZE")
    if override:
        config["batch_size"] = int(override)
    return config


def load_benchmark_configs() -> dict:
    return load_yaml(PROJECT / "configs/benchmarks.yaml")["benchmarks"]


def instantiate_model(name: str):
    config = load_model_configs()[name]
    if config["adapter"] == "llava_ov":
        checkpoint = str(config["checkpoint"])
        if "qwen2.5-vl" in checkpoint.lower():
            raise RuntimeError(
                f"Model config for {name} looks like a Qwen checkpoint ({checkpoint}); "
                "please check configs/models.yaml before running llava_ov jobs."
            )
        return LlavaOVAdapter(config["checkpoint"], config.get("local_path"))
    if config["adapter"] == "qwen25_vl":
        checkpoint = str(config["checkpoint"])
        if "llava" in checkpoint.lower():
            raise RuntimeError(
                f"Model config for {name} looks like a LLaVA checkpoint ({checkpoint}); "
                "please check configs/models.yaml before running qwen25_vl jobs."
            )
        return Qwen25VLAdapter(config["checkpoint"], config.get("local_path"))
    raise RuntimeError(f"Model adapter not implemented yet for {name}")


def instantiate_method(name: str, model):
    if name == "base":
        return BaseMethod(model, resolve_method_config(name))
    if name == "tcd":
        return TCDMethod(model, resolve_method_config(name))
    if name == "dino_heal":
        return DINOHealMethod(model, resolve_method_config(name))
    if name == "season":
        return SeasonMethod(model, resolve_method_config(name))
    raise RuntimeError(f"Method not implemented yet for runnable benchmark path: {name}")


def instantiate_loader(name: str):
    config = load_benchmark_configs()[name]
    if name == "vidhalluc":
        return VidHallucLoader(config["data_root"], config.get("tasks"))
    if name == "videohallucer":
        return VideoHallucerLoader(config["data_root"], config.get("tasks"))
    if name == "eventhallusion":
        return EventHallusionLoader(config["questions_root"], config["video_root"], config.get("tasks"))
    raise RuntimeError(f"Benchmark loader not implemented yet for {name}")


def evaluate_records(benchmark: str, records: list[PredictionRecord]) -> dict:
    if benchmark == "vidhalluc":
        return evaluate_classification(records)
    if benchmark == "videohallucer":
        return pair_accuracy(records)
    if benchmark == "eventhallusion":
        return evaluate_binary(records)
    raise RuntimeError(f"No evaluator for benchmark {benchmark}")


def output_paths(config: dict, model: str, method: str, benchmark: str, task: str | None) -> tuple[Path, Path]:
    raw_root = PROJECT / config["output_dir"]
    summary_root = PROJECT / "results" / "summary"
    manifest_root = PROJECT / config["manifest_dir"]
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    task_suffix = f"__{task}" if task else ""
    stem = f"{config['name']}__{model}__{method}__{benchmark}{task_suffix}"
    return (
        raw_root / f"{stem}.jsonl",
        summary_root / f"{stem}.summary.json",
        manifest_root / f"{stem}.manifest.jsonl",
    )


def count_loader_samples(loader) -> int:
    return sum(1 for _ in loader.iter_samples())


def print_progress(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def emit_record(
    *,
    sample,
    sample_manifest,
    raw_output,
    parse,
    is_correct,
    runtime_seconds: float,
    job: dict,
    model,
    method,
    sampling: dict,
    generation_config: GenerationConfig,
) -> PredictionRecord:
    method_diagnostics = dict(raw_output.diagnostics)
    negative_positions = method_diagnostics.get("negative_frame_positions")
    if negative_positions is not None:
        method_diagnostics["negative_frame_indices"] = [
            sample_manifest.frame_indices[position]
            for position in negative_positions
        ]
    return PredictionRecord(
        sample_id=sample.sample_id,
        model=job["model"],
        method=job["method"],
        benchmark=sample.benchmark,
        task=sample.task,
        prompt=sample.prompt,
        frame_indices=sample_manifest.frame_indices,
        raw_output=raw_output.text,
        normalized_output=parse.value,
        ground_truth=sample.ground_truth,
        is_correct=is_correct,
        parser_status=parse.status,
        error=parse.error,
        model_checkpoint=model.checkpoint,
        method_config=getattr(method, "config", {}) or {},
        sampling_config=sampling,
        generation_config=generation_config.__dict__,
        runtime_seconds=runtime_seconds,
        metadata={**sample.metadata, "manifest": sample_manifest.to_dict(), **method_diagnostics},
    )


def emit_failure_record(
    *,
    sample,
    error: Exception,
    stage: str,
    job: dict,
    model,
    method,
    sampling: dict,
    generation_config: GenerationConfig,
    frame_indices: list[int] | None = None,
    manifest: dict | None = None,
) -> PredictionRecord:
    failure_metadata = {
        **sample.metadata,
        "failure_stage": stage,
        "manifest": manifest,
        "exception_class": type(error).__name__,
        "exception_message": str(error),
    }
    if DEBUG_ERRORS:
        failure_metadata["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    return PredictionRecord(
        sample_id=sample.sample_id,
        model=job["model"],
        method=job["method"],
        benchmark=sample.benchmark,
        task=sample.task,
        prompt=sample.prompt,
        frame_indices=frame_indices or [],
        raw_output="",
        normalized_output=None,
        ground_truth=sample.ground_truth,
        is_correct=None,
        parser_status="missing",
        error=f"{stage}: {error!r}",
        model_checkpoint=model.checkpoint,
        method_config=getattr(method, "config", {}) or {},
        sampling_config=sampling,
        generation_config=generation_config.__dict__,
        metadata=failure_metadata,
    )


def persist_record(record: PredictionRecord, predictions_path: Path, manifest_path: Path) -> None:
    append_jsonl(predictions_path, record)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "sample_id": record.sample_id,
            "benchmark": record.benchmark,
            "task": record.task,
            "manifest": record.metadata.get("manifest"),
            "error": record.error,
        }, ensure_ascii=False, sort_keys=True) + "\n")


def flush_batch(
    *,
    batch_samples,
    batch_frames,
    batch_manifests,
    method,
    generation_config: GenerationConfig,
    predictions_path: Path,
    manifest_path: Path,
    job: dict,
    model,
    sampling: dict,
    seen: int,
    skipped: int,
    total_samples: int,
    records: list[PredictionRecord],
) -> int:
    if not batch_samples:
        return seen

    started = time.perf_counter()
    try:
        raw_outputs = method.generate_batch(
            [item["frames"] for item in batch_frames],
            [item.prompt for item in batch_samples],
            generation_config,
        )
        if len(raw_outputs) != len(batch_samples):
            raise RuntimeError(
                f"generate_batch returned {len(raw_outputs)} outputs for {len(batch_samples)} samples"
            )
    except Exception as exc:
        if DEBUG_ERRORS:
            first_sample = batch_samples[0]
            print(
                "\nSEASON/benchmark generation failed\n"
                f"model={job['model']}\nmethod={job['method']}\n"
                f"benchmark={job['benchmark']}\ntask={job.get('task')}\n"
                f"sample={first_sample.sample_id}\n"
                f"exception={type(exc).__name__}\nmessage={exc}\n"
                f"traceback:\n{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
        for sample, sample_manifest in zip(batch_samples, batch_manifests):
            record = emit_failure_record(
                sample=sample,
                error=exc,
                stage="generation",
                job=job,
                model=model,
                method=method,
                sampling=sampling,
                generation_config=generation_config,
                frame_indices=sample_manifest.frame_indices,
                manifest=sample_manifest.to_dict(),
            )
            persist_record(record, predictions_path, manifest_path)
            records.append(record)
            seen += 1
            print_progress(
                f"FAIL  {job['benchmark']} {skipped + seen}/{total_samples} | new={seen} "
                f"| sample={sample.sample_id} | stage=generation"
            )
        return seen
    batch_runtime = time.perf_counter() - started
    per_sample_runtime = batch_runtime / max(len(batch_samples), 1)

    for sample, sample_manifest, raw_output in zip(batch_samples, batch_manifests, raw_outputs):
        parse = normalize_prediction(sample, raw_output.text)
        is_correct = None if parse.value is None else parse.value == sample.ground_truth
        record = emit_record(
            sample=sample,
            sample_manifest=sample_manifest,
            raw_output=raw_output,
            parse=parse,
            is_correct=is_correct,
            runtime_seconds=per_sample_runtime,
            job=job,
            model=model,
            method=method,
            sampling=sampling,
            generation_config=generation_config,
        )
        persist_record(record, predictions_path, manifest_path)
        records.append(record)
        seen += 1
        processed = skipped + seen
        print_progress(
            f"DONE  {job['benchmark']} {processed}/{total_samples} | new={seen} | sample={sample.sample_id} "
            f"| parse={record.parser_status} | correct={record.is_correct} | batch={len(batch_samples)}"
        )
    return seen


def group_samples_by_task(loader) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for sample in loader.iter_samples():
        grouped[sample.task].append(sample)
    return grouped


def flatten_metrics(prefix: str, value, output: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_metrics(f"{prefix}.{key}" if prefix else key, child, output)
    else:
        output[prefix] = value


def write_metrics_bundle(config: dict) -> dict[str, str]:
    raw_root = PROJECT / config["output_dir"]
    metrics_root = PROJECT / "results" / "metrics"
    tables_root = PROJECT / "results" / "tables"
    metrics_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for path in sorted(raw_root.glob(f"{config['name']}__*.jsonl")):
        for record in read_jsonl(path):
            grouped[(record.model, record.method, record.benchmark)].append(record)

    metrics = {}
    for (model, method, benchmark), records in grouped.items():
        key = f"{model}/{method}/{benchmark}"
        metrics[key] = evaluate_records(benchmark, records)

    metrics_path = metrics_root / f"{config['name']}.metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for result_key, value in metrics.items():
        row = {"result_key": result_key}
        flatten_metrics("", value, row)
        rows.append(row)
    columns = sorted({key for row in rows for key in row}) if rows else ["result_key"]

    csv_path = tables_root / f"{config['name']}.results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    md_path = tables_root / f"{config['name']}.results.md"
    md_lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    md_lines.extend("| " + " | ".join(str(row.get(column, "N/A")) for column in columns) + " |" for row in rows)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "metrics_json": str(metrics_path),
        "results_csv": str(csv_path),
        "results_md": str(md_path),
    }


def run_job(
    config: dict,
    job: dict,
    generation_config: GenerationConfig,
    samples: list | None = None,
    limit: int | None = None,
) -> dict:
    model = instantiate_model(job["model"])
    method = instantiate_method(job["method"], model)
    loader = instantiate_loader(job["benchmark"])
    task = job.get("task")
    if samples is None:
        samples = list(loader.iter_samples())
    elif task is not None:
        samples = [sample for sample in samples if sample.task == task]
    predictions_path, summary_path, manifest_path = output_paths(
        config, job["model"], job["method"], job["benchmark"], task
    )
    resume_keys = valid_resume_keys(predictions_path) if config.get("resume", True) else set()
    records: list[PredictionRecord] = []
    batch_size = max(1, int(getattr(method, "config", {}).get("batch_size", 1)))

    sampling = load_yaml(PROJECT / config["sampling"])
    total_samples = len(samples)
    print_progress(
        f"START {job['model']} / {job['method']} / {job['benchmark']} / {task or 'all'} | total={total_samples} "
        f"| resume_valid={len(resume_keys)} | limit={limit if limit is not None else 'full'}"
    )
    seen = 0
    skipped = 0
    batch_samples = []
    batch_frames = []
    batch_manifests = []
    for sample in samples:
        resume_key = (sample.sample_id, job["model"], job["method"], sample.benchmark, sample.task)
        if resume_key in resume_keys:
            skipped += 1
            continue
        try:
            frames, manifest = sample_video(
                sample.video_path,
                num_frames=sampling["num_frames"],
                strategy=sampling["strategy"],
            )
        except Exception as exc:
            record = emit_failure_record(
                sample=sample,
                error=exc,
                stage="video_sampling",
                job=job,
                model=model,
                method=method,
                sampling=sampling,
                generation_config=generation_config,
            )
            persist_record(record, predictions_path, manifest_path)
            records.append(record)
            seen += 1
            print_progress(
                f"FAIL  {job['benchmark']} {skipped + seen}/{total_samples} | new={seen} "
                f"| sample={sample.sample_id} | stage=video_sampling"
            )
            if limit is not None and seen >= limit:
                break
            continue
        batch_samples.append(sample)
        batch_frames.append({"sample_id": sample.sample_id, "frames": frames})
        batch_manifests.append(manifest)
        should_flush = len(batch_samples) >= batch_size
        if limit is not None:
            remaining = limit - seen
            should_flush = should_flush or len(batch_samples) >= remaining
        if should_flush:
            seen = flush_batch(
                batch_samples=batch_samples,
                batch_frames=batch_frames,
                batch_manifests=batch_manifests,
                method=method,
                generation_config=generation_config,
                predictions_path=predictions_path,
                manifest_path=manifest_path,
                job=job,
                model=model,
                sampling=sampling,
                seen=seen,
                skipped=skipped,
                total_samples=total_samples,
                records=records,
            )
            batch_samples = []
            batch_frames = []
            batch_manifests = []
        if limit is not None and seen >= limit:
            break

    seen = flush_batch(
        batch_samples=batch_samples,
        batch_frames=batch_frames,
        batch_manifests=batch_manifests,
        method=method,
        generation_config=generation_config,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        job=job,
        model=model,
        sampling=sampling,
        seen=seen,
        skipped=skipped,
        total_samples=total_samples,
        records=records,
    )

    summary_records = read_jsonl(predictions_path)
    summary = evaluate_records(job["benchmark"], summary_records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_progress(
        f"END   {job['model']} / {job['method']} / {job['benchmark']} / {task or 'all'} | wrote={len(summary_records)} total records"
    )
    return {
        "job": job,
        "predictions": str(predictions_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "n_new_records": len(records),
        "n_total_samples": total_samples,
    }


def main():
    global DEBUG_ERRORS
    args = parse_args()
    DEBUG_ERRORS = args.debug_errors
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT / config_path
    config = load_yaml(config_path)
    plan = build_plan(config, allow_unvalidated=args.allow_unvalidated)
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
    generation_config = GenerationConfig(**config["generation"])
    results = []
    grouped_samples: dict[str, dict[str, list]] = {}
    benchmark_task_overrides: dict[str, list[str] | None] = {}
    for benchmark_entry in config["benchmarks"]:
        if isinstance(benchmark_entry, str):
            benchmark = benchmark_entry
            tasks_override = None
        else:
            benchmark = benchmark_entry["name"]
            tasks_override = benchmark_entry.get("tasks")
        benchmark_task_overrides[benchmark] = tasks_override
        loader = instantiate_loader(benchmark)
        grouped_samples[benchmark] = group_samples_by_task(loader)

    for benchmark_entry in config["benchmarks"]:
        if isinstance(benchmark_entry, str):
            benchmark = benchmark_entry
        else:
            benchmark = benchmark_entry["name"]
        benchmark_tasks = benchmark_task_overrides.get(benchmark) or load_benchmark_configs()[benchmark].get("tasks") or [None]
        for task in benchmark_tasks:
            task_jobs = [job for job in ready if job["benchmark"] == benchmark and job.get("task") == task]
            task_samples = grouped_samples.get(benchmark, {}).get(task, [])
            for job in task_jobs:
                try:
                    results.append(run_job(config, job, generation_config, samples=task_samples, limit=args.limit))
                except Exception as exc:
                    print_progress(
                        f"ERROR {job['model']} / {job['method']} / {job['benchmark']} / {task or 'all'} | {exc!r}"
                    )
                    results.append({"job": job, "error": repr(exc)})
    bundle = write_metrics_bundle(config)
    print_progress(
        f"AGGREGATED metrics -> {bundle['metrics_json']} | table -> {bundle['results_csv']}"
    )
    print(json.dumps({"executed": results, "artifacts": bundle}, indent=2))


if __name__ == "__main__":
    main()
