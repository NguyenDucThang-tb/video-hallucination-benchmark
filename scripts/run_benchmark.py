#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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
from src.methods.tcd import TCDMethod
from src.models import GenerationConfig, LlavaOVAdapter, Qwen25VLAdapter
from src.models.compatibility import check_compatibility
from src.utils.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment1.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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


def load_model_configs() -> dict:
    return load_yaml(PROJECT / "configs/models.yaml")["models"]


def load_method_configs() -> dict:
    return load_yaml(PROJECT / "configs/methods.yaml")["methods"]


def resolve_method_config(name: str) -> dict:
    config = dict(load_method_configs()[name])
    if name == "base":
        override = os.environ.get("BASE_BATCH_SIZE")
        if override:
            config["batch_size"] = int(override)
    return config


def load_benchmark_configs() -> dict:
    return load_yaml(PROJECT / "configs/benchmarks.yaml")["benchmarks"]


def instantiate_model(name: str):
    config = load_model_configs()[name]
    if config["adapter"] == "llava_ov":
        return LlavaOVAdapter(config["checkpoint"], config.get("local_path"))
    if config["adapter"] == "qwen25_vl":
        return Qwen25VLAdapter(config["checkpoint"], config.get("local_path"))
    raise RuntimeError(f"Model adapter not implemented yet for {name}")


def instantiate_method(name: str, model):
    if name == "base":
        return BaseMethod(model, resolve_method_config(name))
    if name == "tcd":
        return TCDMethod(model, resolve_method_config(name))
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


def output_paths(config: dict, model: str, method: str, benchmark: str) -> tuple[Path, Path]:
    raw_root = PROJECT / config["output_dir"]
    summary_root = PROJECT / "results" / "summary"
    manifest_root = PROJECT / config["manifest_dir"]
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    stem = f"{config['name']}__{model}__{method}__{benchmark}"
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
        metadata={**sample.metadata, "manifest": sample_manifest.to_dict(), **raw_output.diagnostics},
    )


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
    raw_outputs = method.generate_batch(
        [item["frames"] for item in batch_frames],
        [item.prompt for item in batch_samples],
        generation_config,
    )
    batch_runtime = time.perf_counter() - started
    per_sample_runtime = batch_runtime / max(len(batch_samples), 1)

    for sample, sample_manifest, raw_output in zip(batch_samples, batch_manifests, raw_outputs):
        parse = normalize_prediction(sample, raw_output.text)
        is_correct = None if parse.value is None else parse.value == sample.ground_truth
        if sample.task == "sth":
            is_correct = None
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
        append_jsonl(predictions_path, record)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "sample_id": sample.sample_id,
                "benchmark": sample.benchmark,
                "task": sample.task,
                "manifest": sample_manifest.to_dict(),
            }, ensure_ascii=False, sort_keys=True) + "\n")
        records.append(record)
        seen += 1
        processed = skipped + seen
        print_progress(
            f"DONE  {job['benchmark']} {processed}/{total_samples} | new={seen} | sample={sample.sample_id} "
            f"| parse={record.parser_status} | correct={record.is_correct} | batch={len(batch_samples)}"
        )
    return seen


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
    for path in sorted(raw_root.glob("*.jsonl")):
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


def run_job(config: dict, job: dict, generation_config: GenerationConfig, limit: int | None = None) -> dict:
    model = instantiate_model(job["model"])
    method = instantiate_method(job["method"], model)
    loader = instantiate_loader(job["benchmark"])
    predictions_path, summary_path, manifest_path = output_paths(
        config, job["model"], job["method"], job["benchmark"]
    )
    resume_keys = valid_resume_keys(predictions_path) if config.get("resume", True) else set()
    records: list[PredictionRecord] = []
    batch_size = max(1, int(getattr(method, "config", {}).get("batch_size", 1)))

    sampling = load_yaml(PROJECT / config["sampling"])
    total_samples = count_loader_samples(loader)
    print_progress(
        f"START {job['model']} / {job['method']} / {job['benchmark']} | total={total_samples} "
        f"| resume_valid={len(resume_keys)} | limit={limit if limit is not None else 'full'}"
    )
    seen = 0
    skipped = 0
    batch_samples = []
    batch_frames = []
    batch_manifests = []
    for sample in loader.iter_samples():
        resume_key = (sample.sample_id, job["model"], job["method"], sample.benchmark, sample.task)
        if resume_key in resume_keys:
            skipped += 1
            continue
        frames, manifest = sample_video(
            sample.video_path,
            num_frames=sampling["num_frames"],
            strategy=sampling["strategy"],
        )
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
        f"END   {job['model']} / {job['method']} / {job['benchmark']} | wrote={len(summary_records)} total records"
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
    generation_config = GenerationConfig(**config["generation"])
    results = []
    for job in ready:
        try:
            results.append(run_job(config, job, generation_config, args.limit))
        except Exception as exc:
            print_progress(f"ERROR {job['model']} / {job['method']} / {job['benchmark']} | {exc!r}")
            results.append({"job": job, "error": repr(exc)})
    bundle = write_metrics_bundle(config)
    print_progress(
        f"AGGREGATED metrics -> {bundle['metrics_json']} | table -> {bundle['results_csv']}"
    )
    print(json.dumps({"executed": results, "artifacts": bundle}, indent=2))


if __name__ == "__main__":
    main()
