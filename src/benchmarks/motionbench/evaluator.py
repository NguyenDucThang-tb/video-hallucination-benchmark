from __future__ import annotations

from collections import defaultdict

from src.data.schema import PredictionRecord
from src.evaluation.records import latest_records


def _expected(records: list[PredictionRecord], key: str) -> int:
    values = {
        int(record.metadata[key])
        for record in records
        if record.metadata.get(key) is not None
    }
    if len(values) > 1:
        raise ValueError(f"Conflicting MotionBench {key}: {sorted(values)}")
    return next(iter(values), len(records))


def _score(records: list[PredictionRecord], expected_key: str) -> dict:
    expected = _expected(records, expected_key)
    public = [r for r in records if r.metadata.get("has_public_ground_truth") is True]
    correct = sum(r.is_correct is True for r in public)
    runtime_errors = sum(r.error is not None for r in records)
    parser_errors = sum(r.error is None and r.parser_status != "valid" for r in public)
    return {
        "n_expected": expected,
        "n_records": len(records),
        "n_missing": max(expected - len(records), 0),
        "n_public_ground_truth": len(public),
        "n_correct": correct,
        "n_runtime_errors": runtime_errors,
        "n_parser_errors": parser_errors,
        "parse_coverage": (
            (len(public) - parser_errors - runtime_errors) / len(public) if public else None
        ),
        "accuracy": correct / expected if expected and public else None,
        "observed_accuracy": correct / len(public) if public else None,
        "status": (
            "complete"
            if len(records) == expected and runtime_errors == 0
            else "incomplete"
        ),
    }


def evaluate_motionbench(records: list[PredictionRecord]) -> dict:
    """Evaluate public dev labels and retain test predictions for submission."""
    records, duplicate_count = latest_records(records)
    by_task: dict[str, list[PredictionRecord]] = defaultdict(list)
    by_category: dict[str, list[PredictionRecord]] = defaultdict(list)
    dev_records = []
    test_records = []
    for record in records:
        by_task[record.task].append(record)
        by_category[str(record.metadata.get("question_category", "unknown"))].append(record)
        if record.metadata.get("has_public_ground_truth") is True:
            dev_records.append(record)
        else:
            test_records.append(record)

    return {
        "tasks": {key: _score(value, "expected_task_records") for key, value in sorted(by_task.items())},
        "categories": {
            key: _score(value, "expected_category_records")
            for key, value in sorted(by_category.items())
        },
        "dev": _score(dev_records, "expected_split_records") if dev_records else None,
        "test": {
            "n_records": len(test_records),
            "n_parsed": sum(r.normalized_output in {"A", "B", "C", "D"} for r in test_records),
            "accuracy": None,
            "status": "requires_leaderboard" if test_records else "not_run",
        },
        "n_duplicate_records_ignored": duplicate_count,
        "metric": "accuracy",
        "protocol_note": (
            "Only the public dev split is scored locally. Test answers are NA and must be "
            "submitted to the official MotionBench leaderboard."
        ),
    }
