from __future__ import annotations

from collections import defaultdict

from src.data.schema import PredictionRecord
from src.evaluation.records import latest_records


def _score_group(records: list[PredictionRecord]) -> dict:
    n = len(records)
    matched = [
        record for record in records
        if record.error is None
        and record.task != "captioning"
        and record.parser_status == "valid"
    ]
    correct = sum(record.is_correct is True for record in records)
    runtime_errors = sum(record.error is not None for record in records)
    needs_judge = sum(
        record.error is None
        and (record.task == "captioning" or record.parser_status == "requires_judge")
        for record in records
    )
    if not n:
        official_status = "missing"
    elif runtime_errors:
        official_status = "runtime_errors"
    elif needs_judge:
        official_status = "requires_llm_judge"
    else:
        official_status = "complete"
    return {
        "n": n,
        "correct_rule": correct,
        "n_rule_matched": len(matched),
        "n_requires_llm_judge": needs_judge,
        "n_runtime_error": runtime_errors,
        "rule_match_rate": len(matched) / n if n else None,
        "rule_only_accuracy": correct / n if n else None,
        "official_accuracy": (
            correct / n if n and needs_judge == 0 and runtime_errors == 0 else None
        ),
        "official_status": official_status,
    }


def evaluate_tempcompass(records: list[PredictionRecord]) -> dict:
    """Score official rules and expose records that need the upstream LLM judge."""
    records, duplicate_count = latest_records(records)
    by_task = defaultdict(list)
    by_aspect = defaultdict(list)
    by_fine_aspect = defaultdict(list)
    for record in records:
        by_task[record.task].append(record)
        aspect = record.metadata.get("temporal_aspect")
        fine_aspect = record.metadata.get("fine_grained_aspect")
        if aspect:
            by_aspect[aspect].append(record)
        if fine_aspect:
            by_fine_aspect[fine_aspect].append(record)

    return {
        "tasks": {key: _score_group(value) for key, value in sorted(by_task.items())},
        "temporal_aspects": {key: _score_group(value) for key, value in sorted(by_aspect.items())},
        "fine_grained_aspects": {
            key: _score_group(value) for key, value in sorted(by_fine_aspect.items())
        },
        "overall": _score_group(records),
        "n_duplicate_records_ignored": duplicate_count,
        "protocol_note": (
            "Official TempCompass uses rule matching followed by an LLM judge for unmatched "
            "answers; captioning always requires that judge. rule_only_accuracy matches the "
            "upstream --disable_llm behavior."
        ),
    }
