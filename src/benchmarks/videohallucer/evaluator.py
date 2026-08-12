from __future__ import annotations

from collections import defaultdict

from src.data.schema import PredictionRecord


def pair_accuracy(records: list[PredictionRecord]) -> dict[str, float | int | None]:
    pairs: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        pair_id = record.metadata.get("pair_id")
        if pair_id:
            pairs[pair_id].append(record)
    valid_pairs = [items for items in pairs.values() if len(items) == 2]
    strict_correct = sum(all(item.is_correct is True for item in items) for items in valid_pairs)
    return {
        "metric": "strict_pair_accuracy",
        "n_pairs": len(valid_pairs),
        "n_missing_pairs": len(pairs) - len(valid_pairs),
        "correct_pairs": strict_correct,
        "value": strict_correct / len(valid_pairs) if valid_pairs else None,
    }
