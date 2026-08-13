from __future__ import annotations

from collections import defaultdict

from src.data.schema import PredictionRecord


def description_reference(split: str, event_info: dict) -> str | None:
    """Fixed evaluator: upstream checks interleave, while data calls it mix."""
    if split in {"mix", "interleave"}:
        return event_info.get("unexpected")
    if split in {"entire", "misleading"}:
        return event_info.get("caption")
    raise ValueError(f"Unknown split: {split}")


def evaluate_binary(records: list[PredictionRecord]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.task].append(record)
    output = {}
    all_valid = []
    for split, items in grouped.items():
        valid = [r for r in items if r.is_correct is not None]
        all_valid.extend(valid)
        output[split] = {
            "n": len(items), "n_valid": len(valid),
            "n_parser_error": sum(r.parser_status != "valid" for r in items),
            "n_missing": sum(r.parser_status == "missing" for r in items),
            "accuracy": sum(r.is_correct is True for r in valid) / len(valid) if valid else None,
        }
    output["overall"] = {
        "n_valid": len(all_valid),
        "accuracy": sum(r.is_correct is True for r in all_valid) / len(all_valid) if all_valid else None,
    }
    return output
