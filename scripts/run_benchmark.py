#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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
from src.models import GenerationConfig, Qwen25VLAdapter
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


def load_benchmark_configs() -> dict:
    return load_yaml(PROJECT / "configs/benchmarks.yaml")["benchmarks"]


def instantiate_model(name: str):
    config = load_model_configs()[name]
    if config["adapter"] == "qwen25_vl":
        return Qwen25VLAdapter(config["checkpoint"], config.get("local_path"))
    raise RuntimeError(f"Model adapter not implemented yet for {name}")


def instantiate_method(name: str, model):
    if name == "base":
        return BaseMethod(model, load_method_configs()[name])
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


def run_job(config: dict, job: dict, generation_config: GenerationConfig, limit: int | None = None) -> dict:
    model = instantiate_model(job["model"])
    method = instantiate_method(job["method"], model)
    loader = instantiate_loader(job["benchmark"])
    predictions_path, summary_path, manifest_path = output_paths(
        config, job["model"], job["method"], job["benchmark"]
    )
    resume_keys = valid_resume_keys(predictions_path) if config.get("resume", True) else set()
    records: list[PredictionRecord] = []

    sampling = load_yaml(PROJECT / config["sampling"])
    seen = 0
    for sample in loader.iter_samples():
        resume_key = (sample.sample_id, job["model"], job["method"], sample.benchmark, sample.task)
        if resume_key in resume_keys:
            continue
        started = time.perf_counter()
        frames, manifest = sample_video(
            sample.video_path,
            num_frames=sampling["num_frames"],
            strategy=sampling["strategy"],
        )
        raw_output = method.generate(frames, sample.prompt, generation_config)
        parse = normalize_prediction(sample, raw_output.text)
        is_correct = None if parse.value is None else parse.value == sample.ground_truth
        if sample.task == "sth":
            is_correct = None
        record = PredictionRecord(
            sample_id=sample.sample_id,
            model=job["model"],
            method=job["method"],
            benchmark=sample.benchmark,
            task=sample.task,
            prompt=sample.prompt,
            frame_indices=manifest.frame_indices,
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
            runtime_seconds=time.perf_counter() - started,
            metadata={**sample.metadata, "manifest": manifest.to_dict(), **raw_output.diagnostics},
        )
        append_jsonl(predictions_path, record)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "sample_id": sample.sample_id,
                "benchmark": sample.benchmark,
                "task": sample.task,
                "manifest": manifest.to_dict(),
            }, ensure_ascii=False, sort_keys=True) + "\n")
        records.append(record)
        seen += 1
        if limit is not None and seen >= limit:
            break

    summary_records = read_jsonl(predictions_path)
    summary = evaluate_records(job["benchmark"], summary_records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "job": job,
        "predictions": str(predictions_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "n_new_records": len(records),
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
            results.append({"job": job, "error": repr(exc)})
    print(json.dumps({"executed": results}, indent=2))


if __name__ == "__main__":
    main()
