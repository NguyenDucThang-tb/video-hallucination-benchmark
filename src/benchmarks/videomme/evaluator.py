from __future__ import annotations

from collections import defaultdict

from src.data.schema import PredictionRecord
from src.evaluation.records import latest_records


def _score(records: list[PredictionRecord]) -> dict:
    n = len(records)
    correct = sum(record.is_correct is True for record in records)
    errors = sum(record.error is not None for record in records)
    invalid = sum(record.error is None and record.parser_status != "valid" for record in records)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "n_runtime_errors": errors,
        "n_invalid_answers": invalid,
        "status": "complete" if n and not errors else "incomplete" if n else "missing",
    }


def evaluate_videomme(records: list[PredictionRecord]) -> dict:
    records, duplicate_count = latest_records(records)
    groups = {key: defaultdict(list) for key in ("duration", "domain", "sub_category", "task_type")}
    for record in records:
        for key, grouped in groups.items():
            if key == "duration":
                continue
            value = record.metadata.get(key) or "unknown"
            grouped[str(value)].append(record)
        groups["duration"][record.task].append(record)
    return {
        "overall": _score(records),
        "duration": {key: _score(value) for key, value in sorted(groups["duration"].items())},
        "domain": {key: _score(value) for key, value in sorted(groups["domain"].items())},
        "sub_category": {key: _score(value) for key, value in sorted(groups["sub_category"].items())},
        "task_type": {key: _score(value) for key, value in sorted(groups["task_type"].items())},
        "n_duplicate_records_ignored": duplicate_count,
        "metric": "multiple-choice accuracy",
        "protocol_note": "Subtitle-free; controlled 8-frame uniform sampling. Not directly comparable to leaderboard runs using different frame budgets or subtitles.",
    }
