#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROJECT = Path(__file__).resolve().parents[1]
TASKS = {
    "orh": ("object_relation", "object_relation.json"),
    "tph": ("temporal", "temporal.json"),
    "sdh": ("semantic_detail", "semantic_detail.json"),
    "efh": ("external_factual", "external_factual.json"),
    "enfh": ("external_nonfactual", "external_nonfactual.json"),
}
PROMPT_SUFFIX = "Answer the question using 'yes' or 'no'."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit VideoHallucer Base reproducibility")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT / "external/VideoHallucer/videohallucer_datasets",
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "results/raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/audit")
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=PROJECT / "external/VideoHallucer/videohallucer_datasets",
        help="Annotation root used only for byte-level provenance comparison",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prompt_hash(prompt: str) -> str:
    return sha256_bytes(prompt.encode("utf-8"))


def annotation_path(root: Path, task: str) -> Path:
    folder, filename = TASKS[task]
    return root / folder / filename


def video_path(root: Path, task: str, filename: str) -> Path:
    folder, _ = TASKS[task]
    return root / folder / "videos" / filename


def load_annotations(root: Path) -> dict[str, list[dict]]:
    annotations = {}
    for task in TASKS:
        source = annotation_path(root, task)
        rows = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError(f"Expected a JSON list in {source}")
        annotations[task] = rows
    return annotations


def build_inventory(
    dataset_root: Path,
    annotations: dict[str, list[dict]],
) -> tuple[list[dict], dict[str, dict]]:
    inventory = []
    summary = {}
    for task, rows in annotations.items():
        pair_signatures = []
        branch_signatures = []
        missing_branches = 0
        missing_videos = set()
        videos = set()
        for pair_index, row in enumerate(rows):
            pair_id = f"{task}:{pair_index}"
            pair_signatures.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
            for branch in ("basic", "hallucination"):
                item = row.get(branch)
                if not isinstance(item, dict):
                    missing_branches += 1
                    continue
                path = video_path(dataset_root, task, str(item.get("video", "")))
                if not path.is_file():
                    missing_videos.add(str(path))
                videos.add(str(item.get("video", "")))
                question = str(item.get("question", ""))
                answer = str(item.get("answer", "")).lower()
                sample_id = f"{pair_id}:{branch}"
                branch_signatures.append((branch, str(item.get("video")), question, answer))
                inventory.append({
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "task": task,
                    "branch": branch,
                    "video_path": str(path),
                    "video_exists": path.is_file(),
                    "question": question,
                    "prompt": f"{question}\n{PROMPT_SUFFIX}",
                    "ground_truth": answer,
                    "source": str(annotation_path(dataset_root, task)),
                })
        summary[task] = {
            "raw_annotation_rows": len(rows),
            "basic_rows": sum(row.get("basic") is not None for row in rows),
            "hallucination_rows": sum(row.get("hallucination") is not None for row in rows),
            "valid_pairs": sum(
                isinstance(row.get("basic"), dict)
                and isinstance(row.get("hallucination"), dict)
                for row in rows
            ),
            "unique_videos": len(videos),
            "missing_videos": len(missing_videos),
            "missing_branches": missing_branches,
            "duplicate_pairs": len(pair_signatures) - len(set(pair_signatures)),
            "duplicate_branch_rows": len(branch_signatures) - len(set(branch_signatures)),
        }
    return inventory, summary


def build_pair_inventory(branch_inventory: list[dict]) -> list[dict]:
    pairs: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in branch_inventory:
        pairs[(row["task"], row["pair_id"])][row["branch"]] = row
    output = []
    for (task, pair_id), branches in sorted(pairs.items()):
        basic = branches.get("basic", {})
        hallucination = branches.get("hallucination", {})
        output.append({
            "task": task,
            "pair_id": pair_id,
            "basic_sample_id": basic.get("sample_id", ""),
            "hallucination_sample_id": hallucination.get("sample_id", ""),
            "video_basic": basic.get("video_path", ""),
            "video_hallucination": hallucination.get("video_path", ""),
            "ground_truth_basic": basic.get("ground_truth", ""),
            "ground_truth_hallucination": hallucination.get("ground_truth", ""),
            "pair_valid": set(branches) == {"basic", "hallucination"},
        })
    return output


def dataset_inventory_rows(summary: dict[str, dict], revision: str) -> list[dict]:
    return [
        {
            "task": task.upper(),
            "annotation_rows": values["raw_annotation_rows"],
            "basic": values["basic_rows"],
            "hallucination": values["hallucination_rows"],
            "valid_pairs": values["valid_pairs"],
            "missing_videos": values["missing_videos"],
            "missing_branches": values["missing_branches"],
            "dataset_revision": revision,
        }
        for task, values in summary.items()
    ]


def read_jsonl_files(raw_dir: Path) -> tuple[list[dict], list[str]]:
    records = []
    errors = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: {exc}")
                    continue
                if row.get("benchmark") == "videohallucer" and row.get("method") == "base":
                    row["_source_file"] = str(path)
                    row["_source_line"] = line_number
                    records.append(row)
    return records, errors


def official_value(raw_output: str, expected: str) -> str | None:
    pattern = rf"\b({re.escape(expected.strip())})\b"
    return expected.strip().lower() if re.search(pattern, raw_output, re.IGNORECASE) else None


def strict_local_value(raw_output: str) -> tuple[str | None, str]:
    matches = list(dict.fromkeys(x.lower() for x in re.findall(r"\b(yes|no)\b", raw_output, re.I)))
    if not matches:
        return None, "unparseable"
    if len(matches) > 1:
        return None, "ambiguous"
    return matches[0], "valid"


def latest_base_records(records: Iterable[dict]) -> tuple[dict[tuple[str, str], dict], Counter]:
    latest = {}
    counts = Counter()
    for row in records:
        key = (str(row.get("model", "")), str(row.get("sample_id", "")))
        latest[key] = row
        counts[key] += 1
    return latest, counts


def create_record_audit(
    records: list[dict],
    inventory: list[dict],
) -> tuple[list[dict], dict[tuple[str, str], dict], Counter]:
    expected = {row["sample_id"]: row for row in inventory}
    latest, counts = latest_base_records(records)
    audit_rows = []
    for (model, sample_id), row in sorted(latest.items()):
        metadata = row.get("metadata") or {}
        manifest = metadata.get("manifest") or {}
        expected_row = expected.get(sample_id)
        stored_prompt = str(row.get("prompt", ""))
        raw_output = str(row.get("raw_output", ""))
        strict_value, strict_status = strict_local_value(raw_output)
        path = str(manifest.get("video_path") or (expected_row or {}).get("video_path", ""))
        audit_rows.append({
            "model": model,
            "task": row.get("task", ""),
            "pair_id": metadata.get("pair_id", ""),
            "branch": metadata.get("branch", ""),
            "sample_id": sample_id,
            "video_path": path,
            "video_exists": Path(path).is_file() if path else False,
            "frame_indices": json.dumps(row.get("frame_indices", [])),
            "prompt_hash": prompt_hash(stored_prompt),
            "prompt_matches_annotation": bool(expected_row and stored_prompt == expected_row["prompt"]),
            "raw_output": raw_output,
            "parsed_output": row.get("normalized_output"),
            "official_parsed_output": official_value(raw_output, str(row.get("ground_truth", ""))),
            "strict_local_parsed_output": strict_value,
            "ground_truth": row.get("ground_truth"),
            "parser_status": row.get("parser_status"),
            "strict_local_parser_status": strict_status,
            "is_correct": row.get("is_correct"),
            "error": row.get("error"),
            "duplicate_records": counts[(model, sample_id)] - 1,
            "source_file": row.get("_source_file"),
            "source_line": row.get("_source_line"),
        })
    return audit_rows, latest, counts


def pair_metric_rows(
    inventory: list[dict],
    latest: dict[tuple[str, str], dict],
    counts: Counter,
) -> list[dict]:
    models = sorted({model for model, _ in latest})
    expected_by_task = defaultdict(list)
    for row in inventory:
        expected_by_task[row["task"]].append(row)
    output = []
    for model in models:
        task_rows = []
        for task in TASKS:
            expected_rows = expected_by_task[task]
            if not expected_rows:
                continue
            pairs = defaultdict(dict)
            official_branch_correct = 0
            strict_branch_correct = 0
            missing_records = 0
            parser_errors = 0
            duplicate_records = 0
            for expected in expected_rows:
                record = latest.get((model, expected["sample_id"]))
                if record is None:
                    missing_records += 1
                    continue
                duplicate_records += counts[(model, expected["sample_id"])] - 1
                raw_output = str(record.get("raw_output", ""))
                official_ok = official_value(raw_output, expected["ground_truth"]) is not None
                strict_value, strict_status = strict_local_value(raw_output)
                strict_ok = strict_status == "valid" and strict_value == expected["ground_truth"]
                official_branch_correct += int(official_ok)
                strict_branch_correct += int(strict_ok)
                parser_errors += int(record.get("parser_status") != "valid")
                pairs[expected["pair_id"]][expected["branch"]] = (official_ok, strict_ok)

            expected_pairs = len(expected_rows) // 2
            observed_pairs = len(pairs)
            complete_pairs = sum(set(branches) == {"basic", "hallucination"} for branches in pairs.values())
            official_pair_correct = sum(
                set(branches) == {"basic", "hallucination"}
                and all(result[0] for result in branches.values())
                for branches in pairs.values()
            )
            strict_pair_correct = sum(
                set(branches) == {"basic", "hallucination"}
                and all(result[1] for result in branches.values())
                for branches in pairs.values()
            )
            row = {
                "model": model,
                "task": task,
                "expected_branches": len(expected_rows),
                "expected_pairs": expected_pairs,
                "observed_pairs": observed_pairs,
                "complete_pairs": complete_pairs,
                "missing_records": missing_records,
                "duplicate_records": duplicate_records,
                "parser_error_records": parser_errors,
                "branch_accuracy": official_branch_correct / len(expected_rows),
                "official_compatible_strict_pair_accuracy": official_pair_correct / expected_pairs,
                "strict_local_strict_pair_accuracy": strict_pair_correct / expected_pairs,
                "local_current_pair_accuracy": (
                    official_pair_correct / observed_pairs if observed_pairs else None
                ),
                "official_correct_pairs": official_pair_correct,
                "strict_local_correct_pairs": strict_pair_correct,
            }
            output.append(row)
            task_rows.append(row)
        if not task_rows:
            continue
        official_values = [row["official_compatible_strict_pair_accuracy"] for row in task_rows]
        strict_values = [row["strict_local_strict_pair_accuracy"] for row in task_rows]
        total_pairs = sum(row["expected_pairs"] for row in task_rows)
        output.extend([
            {
                "model": model,
                "task": "AVG_MACRO",
                "expected_branches": sum(row["expected_branches"] for row in task_rows),
                "expected_pairs": total_pairs,
                "official_compatible_strict_pair_accuracy": sum(official_values) / len(official_values),
                "strict_local_strict_pair_accuracy": sum(strict_values) / len(strict_values),
            },
            {
                "model": model,
                "task": "AVG_POOLED",
                "expected_branches": sum(row["expected_branches"] for row in task_rows),
                "expected_pairs": total_pairs,
                "official_compatible_strict_pair_accuracy": (
                    sum(row["official_correct_pairs"] for row in task_rows) / total_pairs
                ),
                "strict_local_strict_pair_accuracy": (
                    sum(row["strict_local_correct_pairs"] for row in task_rows) / total_pairs
                ),
            },
        ])
    return output


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def provenance_rows(dataset_root: Path, upstream_root: Path) -> list[dict]:
    rows = []
    for task in TASKS:
        local = annotation_path(dataset_root, task)
        upstream = annotation_path(upstream_root, task)
        rows.append({
            "task": task,
            "dataset_annotation": str(local),
            "dataset_sha256": sha256_bytes(local.read_bytes()),
            "upstream_annotation": str(upstream),
            "upstream_sha256": sha256_bytes(upstream.read_bytes()),
            "byte_identical": local.read_bytes() == upstream.read_bytes(),
        })
    return rows


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.dataset_root)
    inventory, summary = build_inventory(args.dataset_root, annotations)
    pair_inventory = build_pair_inventory(inventory)
    records, jsonl_errors = read_jsonl_files(args.raw_dir)
    audit_rows, latest, counts = create_record_audit(records, inventory)
    metric_rows = pair_metric_rows(inventory, latest, counts)

    write_csv(
        args.output_dir / "videohallucer_pair_inventory.csv",
        pair_inventory,
        [
            "task", "pair_id", "basic_sample_id", "hallucination_sample_id",
            "video_basic", "video_hallucination", "ground_truth_basic",
            "ground_truth_hallucination", "pair_valid",
        ],
    )
    write_csv(
        args.output_dir / "videohallucer_branch_inventory.csv",
        inventory,
        [
            "sample_id", "pair_id", "task", "branch", "video_path", "video_exists",
            "question", "prompt", "ground_truth", "source",
        ],
    )
    write_csv(
        args.output_dir / "videohallucer_dataset_inventory.csv",
        dataset_inventory_rows(summary, "current-upstream-8b785d1"),
        [
            "task", "annotation_rows", "basic", "hallucination", "valid_pairs",
            "missing_videos", "missing_branches", "dataset_revision",
        ],
    )
    write_csv(
        args.output_dir / "base_record_audit.csv",
        audit_rows,
        [
            "model", "task", "pair_id", "branch", "sample_id", "video_path",
            "video_exists", "frame_indices", "prompt_hash", "prompt_matches_annotation",
            "raw_output", "parsed_output", "official_parsed_output",
            "strict_local_parsed_output", "ground_truth", "parser_status",
            "strict_local_parser_status", "is_correct", "error", "duplicate_records",
            "source_file", "source_line",
        ],
    )
    write_csv(
        args.output_dir / "base_metric_comparison.csv",
        metric_rows,
        [
            "model", "task", "expected_branches", "expected_pairs", "observed_pairs",
            "complete_pairs", "missing_records", "duplicate_records", "parser_error_records",
            "branch_accuracy", "official_compatible_strict_pair_accuracy",
            "strict_local_strict_pair_accuracy", "local_current_pair_accuracy",
            "official_correct_pairs", "strict_local_correct_pairs",
        ],
    )
    write_csv(
        args.output_dir / "videohallucer_annotation_provenance.csv",
        provenance_rows(args.dataset_root, args.upstream_root),
        [
            "task", "dataset_annotation", "dataset_sha256", "upstream_annotation",
            "upstream_sha256", "byte_identical",
        ],
    )
    (args.output_dir / "videohallucer_dataset_summary.json").write_text(
        json.dumps({"tasks": summary, "jsonl_errors": jsonl_errors}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote VideoHallucer audit artifacts to {args.output_dir}")
    print(f"Inventory records: {len(inventory)}; Base records audited: {len(audit_rows)}")


if __name__ == "__main__":
    main()
