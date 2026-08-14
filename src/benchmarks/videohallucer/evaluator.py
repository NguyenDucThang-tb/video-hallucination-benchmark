from __future__ import annotations

from collections import defaultdict

from src.data.schema import PredictionRecord
from src.evaluation.metrics import macro_average


def _pair_stats(records: list[PredictionRecord]) -> dict[str, float | int | None]:
    pairs: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        pair_id = record.metadata.get("pair_id")
        if pair_id:
            pairs[pair_id].append(record)
    valid_pairs = [items for items in pairs.values() if len(items) == 2]
    strict_correct = sum(all(item.is_correct is True for item in items) for items in valid_pairs)
    return {
        "n_pairs": len(valid_pairs),
        "n_missing_pairs": len(pairs) - len(valid_pairs),
        "correct_pairs": strict_correct,
        "accuracy": strict_correct / len(valid_pairs) if valid_pairs else None,
    }


def pair_accuracy(records: list[PredictionRecord]) -> dict[str, dict | float | int | str | None]:
    per_task = {}
    for task in ("orh", "tph", "sdh", "efh", "enfh"):
        per_task[task] = _pair_stats([record for record in records if record.task == task])

    per_task["avg"] = {
        "accuracy": macro_average({
            task: stats["accuracy"]
            for task, stats in per_task.items()
            if isinstance(stats, dict) and "accuracy" in stats
        })
    }
    per_task["metric"] = "strict_pair_accuracy"
    return per_task
