#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.vidhalluc.evaluator import evaluate_classification
from src.benchmarks.vidhalluc.loader import VIDEO_SUFFIXES
from src.data.jsonl import read_jsonl
from src.evaluation.parsers import (
    parse_ab_ba,
    parse_vidhalluc_sth,
    parse_vidhalluc_tsh_official,
)
from src.evaluation.records import latest_records
from src.utils.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Audit existing VidHalluc TSH/STH artifacts without inference")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--output-dir", default="results/audit")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--simcse-model", default=None, help="Local SimCSE checkpoint; omitted means official STH=N/A")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_records(raw_dir: Path, experiment: str | None):
    records = []
    paths = []
    for path in sorted(raw_dir.glob("*__vidhalluc__*.jsonl")):
        if experiment and not path.name.startswith(f"{experiment}__"):
            continue
        path_records = [record for record in read_jsonl(path) if record.task in {"tsh", "sth"}]
        if path_records:
            paths.append(path)
            records.extend(path_records)
    return records, paths


def inventory_rows() -> tuple[list[dict], dict]:
    benchmark = load_yaml(PROJECT / "configs/benchmarks.yaml")["benchmarks"]["vidhalluc"]
    root = Path(benchmark["data_root"])
    status = {"dataset_root": str(root), "available": root.exists(), "errors": []}
    annotation_root = root if (root / "tsh.json").exists() else root.parent
    candidates = defaultdict(list)
    if root.exists():
        for path in root.rglob("*"):
            if path.suffix.lower() in VIDEO_SUFFIXES:
                candidates[path.name].append(path)
                candidates[path.stem].append(path)

    rows = []
    duplicate_annotation_ids = {}
    for task, filename in (("tsh", "tsh.json"), ("sth", "sth.json")):
        path = annotation_root / filename
        if not path.exists():
            status["errors"].append(f"FileNotFoundError: {path}")
            continue
        raw_pairs = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=lambda pairs: pairs)
        top_keys = [str(key) for key, _ in raw_pairs]
        duplicate_annotation_ids[task] = sorted(key for key, count in Counter(top_keys).items() if count > 1)
        data = json.loads(path.read_text(encoding="utf-8"))
        for annotation_id, item in data.items():
            video_id = item.get("video") if task == "tsh" else annotation_id
            matches = candidates.get(str(video_id), []) + candidates.get(f"{video_id}.mp4", [])
            unique_matches = list(dict.fromkeys(matches))
            if len(unique_matches) == 1:
                mapping_status = "resolved"
                resolved_path = str(unique_matches[0])
            elif len(unique_matches) > 1:
                mapping_status = "ambiguous"
                resolved_path = "|".join(str(match) for match in unique_matches)
            else:
                mapping_status = "missing"
                resolved_path = ""
            ground_truth = (
                str(item.get("Correct Answer", "")).strip().upper()
                if task == "tsh"
                else str(item.get("Scene change", "")).strip().lower()
            )
            valid_label = ground_truth in ({"AB", "BA"} if task == "tsh" else {"yes", "no"})
            reasons = []
            if mapping_status != "resolved":
                reasons.append("video_" + mapping_status)
            if not valid_label:
                reasons.append("invalid_label")
            rows.append({
                "task": task,
                "sample_id": f"{task}:{annotation_id}",
                "annotation_video_id": video_id,
                "resolved_video_path": resolved_path,
                "mapping_status": mapping_status,
                "ground_truth": ground_truth,
                "skip_reason": ";".join(reasons),
                "evaluated": False,
            })
    status["counts"] = dict(Counter(row["task"] for row in rows))
    status["unique_video_ids"] = {
        task: len({row["annotation_video_id"] for row in rows if row["task"] == task})
        for task in ("tsh", "sth")
    }
    status["duplicate_annotation_ids"] = duplicate_annotation_ids
    status["missing_or_ambiguous_videos"] = sum(row["mapping_status"] != "resolved" for row in rows)
    status["invalid_labels"] = sum("invalid_label" in row["skip_reason"] for row in rows)
    return rows, status


def mark_evaluated(inventory: list[dict], records) -> None:
    evaluated = {(record.task, record.sample_id) for record in records}
    for row in inventory:
        row["evaluated"] = (row["task"], row["sample_id"]) in evaluated


def tsh_reparse_rows(records) -> list[dict]:
    rows = []
    for record in records:
        if record.task != "tsh":
            continue
        official = parse_vidhalluc_tsh_official(record.raw_output)
        diagnostic = parse_ab_ba(record.raw_output)
        correct = None if official.value is None else official.value == str(record.ground_truth).upper()
        rows.append({
            "sample_id": record.sample_id,
            "model": record.model,
            "method": record.method,
            "ground_truth": record.ground_truth,
            "raw_model_output": record.raw_output,
            "local_prediction": record.normalized_output,
            "official_prediction": official.value,
            "diagnostic_prediction": diagnostic.value,
            "parse_status": official.status,
            "correct_status": correct,
            "parser_difference": record.normalized_output != official.value,
            "runtime_error": record.error,
        })
    return rows


def rendered_prompt_rows(records) -> list[dict]:
    rows = []
    for record in records:
        if record.task != "tsh":
            continue
        branch = record.metadata.get("original_branch") or record.metadata.get("branch_diagnostics", {}).get("original", {})
        rendered = record.metadata.get("rendered_prompt") or branch.get("rendered_prompt")
        vision_supplied = record.metadata.get("vision_tensor_supplied")
        if vision_supplied is None:
            vision_supplied = branch.get("vision_tensor_supplied")
        input_keys = record.metadata.get("model_input_keys") or branch.get("model_input_keys")
        rows.append({
            "sample_id": record.sample_id,
            "model": record.model,
            "method": record.method,
            "source_prompt": record.prompt,
            "rendered_prompt": rendered,
            "render_status": "captured" if rendered else "not_captured_in_legacy_record",
            "vision_tensor_supplied": vision_supplied,
            "model_input_keys": input_keys,
        })
    return rows


def group_metrics(records) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[f"{record.model}/{record.method}"].append(record)
    return {key: evaluate_classification(items) for key, items in sorted(grouped.items())}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def run_simcse(records, checkpoint: str) -> dict:
    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        return {"official_status": "SIMCSE_NOT_AVAILABLE", "reason": str(exc)}

    path = Path(checkpoint).expanduser()
    local_only = path.exists()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=local_only)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=local_only).eval()
    sentences = set()
    parsed_items = []
    for record in records:
        if record.task != "sth":
            continue
        parsed, locations = parse_vidhalluc_sth(record.raw_output)
        gt_locations = str(record.metadata.get("locations", ""))
        parsed_items.append((record, parsed.value, locations or "", gt_locations))
        if parsed.value == "yes" and str(record.ground_truth).lower() == "yes":
            for description in (locations or "", gt_locations):
                match = __import__("re").match(r"from (.+?) to (.+?)\.", description, flags=__import__("re").IGNORECASE)
                if match:
                    sentences.update((match.group(1).strip(), match.group(2).strip()))

    embeddings = {}
    ordered = sorted(sentences)
    for start in range(0, len(ordered), 32):
        batch = ordered[start:start + 32]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            vectors = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        for sentence, vector in zip(batch, vectors):
            embeddings[sentence] = vector

    def scene_score(left: str, right: str) -> float:
        a, b = embeddings[left], embeddings[right]
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if similarity <= 0.5:
            return 0.0
        return (_sigmoid(similarity) - _sigmoid(0.5)) / (_sigmoid(1.0) - _sigmoid(0.5))

    truth = []
    predictions = []
    total_description = 0.0
    max_description = 0.0
    import re
    for record, prediction, locations, gt_locations in parsed_items:
        gt_yes = str(record.ground_truth).lower() == "yes"
        pred_yes = prediction == "yes"
        truth.append(gt_yes)
        predictions.append(pred_yes)
        if gt_yes and pred_yes:
            model_match = re.match(r"from (.+?) to (.+?)\.", locations, flags=re.IGNORECASE)
            gt_match = re.match(r"from (.+?) to (.+?)\.", gt_locations, flags=re.IGNORECASE)
            if model_match and gt_match:
                total_description += scene_score(model_match.group(1).strip(), gt_match.group(1).strip())
                total_description += scene_score(model_match.group(2).strip(), gt_match.group(2).strip())
            max_description += 2.0

    tp = sum(gt and pred for gt, pred in zip(truth, predictions))
    tn = sum(not gt and not pred for gt, pred in zip(truth, predictions))
    fp = sum(not gt and pred for gt, pred in zip(truth, predictions))
    fn = sum(gt and not pred for gt, pred in zip(truth, predictions))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    classification = ((mcc + 1.0) / 2.0) ** 2
    description = total_description / max_description if max_description else 0.0
    return {
        "official_status": "VERIFIED",
        "simcse_checkpoint": checkpoint,
        "similarity_threshold_low": 0.5,
        "mcc": mcc,
        "classification_score": classification,
        "description_accuracy": description,
        "overall_score": 0.6 * classification + 0.4 * description,
        "n": len(parsed_items),
    }


def main():
    args = parse_args()
    raw_dir = (PROJECT / args.raw_dir).resolve()
    output_dir = (PROJECT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, paths = collect_records(raw_dir, args.experiment)
    records, duplicate_count = latest_records(records)
    inventory, inventory_status = inventory_rows()
    mark_evaluated(inventory, records)
    reparse = tsh_reparse_rows(records)

    write_csv(output_dir / "vidhalluc_tsh_sth_inventory.csv", inventory, [
        "task", "sample_id", "annotation_video_id", "resolved_video_path",
        "mapping_status", "ground_truth", "skip_reason", "evaluated",
    ])
    write_csv(output_dir / "vidhalluc_tsh_reparse.csv", reparse, [
        "sample_id", "model", "method", "ground_truth", "raw_model_output",
        "local_prediction", "official_prediction", "diagnostic_prediction",
        "parse_status", "correct_status", "parser_difference", "runtime_error",
    ])
    with (output_dir / "tsh_rendered_prompts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rendered_prompt_rows(records):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = group_metrics(records)
    sth = {
        "official_status": "SIMCSE_NOT_AVAILABLE",
        "reason": "Pass --simcse-model with the official checkpoint to execute description scoring.",
        "diagnostic_metrics_by_model_method": {
            key: value["sth"] for key, value in metrics.items()
        },
    }
    if args.simcse_model:
        sth["official_runs"] = {
            key: run_simcse(items, args.simcse_model)
            for key, items in _group_record_lists(records).items()
        }
        sth["official_status"] = "EXECUTED"
    (output_dir / "vidhalluc_sth_metrics.json").write_text(
        json.dumps(sth, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "raw_files": [str(path) for path in paths],
        "records_after_deduplication": len(records),
        "duplicate_records_ignored": duplicate_count,
        "inventory": inventory_status,
        "metrics_by_model_method": metrics,
    }
    (output_dir / "vidhalluc_tsh_sth_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote VidHalluc TSH/STH audit artifacts to {output_dir}")


def _group_record_lists(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[f"{record.model}/{record.method}"].append(record)
    return grouped


if __name__ == "__main__":
    main()
