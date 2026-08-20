from __future__ import annotations

from collections import defaultdict

from src.data.schema import PredictionRecord
from src.evaluation.records import latest_records


def description_reference(split: str, event_info: dict) -> str | None:
    """Fixed evaluator: upstream checks interleave, while data calls it mix."""
    if split in {"mix", "interleave"}:
        return event_info.get("unexpected")
    if split in {"entire", "misleading"}:
        return event_info.get("caption")
    raise ValueError(f"Unknown split: {split}")


def evaluate_binary(records: list[PredictionRecord]) -> dict[str, dict]:
    records, duplicate_count = latest_records(records)
    grouped = defaultdict(list)
    for record in records:
        grouped[record.task].append(record)
    output = {}
    for split in ("entire", "misleading", "mix"):
        items = grouped.get(split, [])
        valid = [r for r in items if r.is_correct is not None]
        output[split] = {
            "n": len(items), "n_valid": len(valid),
            "correct": sum(r.is_correct is True for r in items),
            "n_parser_error": sum(r.parser_status != "valid" for r in items),
            "n_missing": sum(r.parser_status == "missing" or bool(r.error) for r in items),
            "accuracy": sum(r.is_correct is True for r in items) / len(items) if items else None,
        }
    output["overall"] = {
        "n_valid": sum(1 for r in records if r.is_correct is not None),
        "n": len(records),
        "correct": sum(r.is_correct is True for r in records),
        "n_parser_error": sum(r.parser_status != "valid" for r in records),
        "n_missing": sum(r.parser_status == "missing" or bool(r.error) for r in records),
        "accuracy": sum(r.is_correct is True for r in records) / len(records) if records else None,
    }
    output["description"] = {
        "accuracy": None,
        "status": "N/A",
        "reason": "GPT-4o description judging was not supplied; binary evaluation does not require an API.",
    }
    output["n_duplicate_records_ignored"] = duplicate_count
    return output
