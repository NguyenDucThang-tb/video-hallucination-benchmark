from __future__ import annotations

from collections import defaultdict
import re

from src.data.schema import PredictionRecord
from src.evaluation.metrics import strict_macro_average
from src.evaluation.records import latest_records


def _pair_stats(records: list[PredictionRecord]) -> dict[str, float | int | None]:
    pairs: dict[str, dict[str, PredictionRecord]] = defaultdict(dict)
    for record in records:
        pair_id = record.metadata.get("pair_id")
        branch = record.metadata.get("branch")
        if pair_id and branch in {"basic", "hallucination"}:
            pairs[str(pair_id)][str(branch)] = record

    def official_hit(record: PredictionRecord) -> bool:
        expected = re.escape(str(record.ground_truth).strip())
        return re.search(rf"\b({expected})\b", record.raw_output, re.IGNORECASE) is not None

    complete = [items for items in pairs.values() if set(items) == {"basic", "hallucination"}]
    strict_correct = sum(all(official_hit(item) for item in items.values()) for items in complete)
    declared_counts = {
        int(record.metadata["expected_task_pairs"])
        for record in records
        if record.metadata.get("expected_task_pairs") is not None
    }
    if len(declared_counts) > 1:
        raise ValueError(f"Conflicting expected_task_pairs metadata: {sorted(declared_counts)}")
    has_annotation_denominator = bool(declared_counts)
    expected_pairs = next(iter(declared_counts), len(pairs))
    if len(pairs) > expected_pairs:
        raise ValueError(
            f"Observed {len(pairs)} VideoHallucer pairs, more than declared {expected_pairs}"
        )
    return {
        "n_pairs": expected_pairs,
        "n_observed_pairs": len(pairs),
        "n_complete_pairs": len(complete),
        "n_missing_pairs": expected_pairs - len(complete),
        "correct_pairs": strict_correct,
        "n_parser_error_records": sum(r.parser_status != "valid" for r in records),
        "denominator_source": "annotation_metadata" if declared_counts else "unverified_observed_records",
        "protocol_status": "VALID" if has_annotation_denominator else "UNVERIFIED_DENOMINATOR",
        "observed_only_accuracy": strict_correct / len(pairs) if pairs else None,
        "accuracy": (
            strict_correct / expected_pairs
            if has_annotation_denominator and expected_pairs
            else None
        ),
    }


def pair_accuracy(records: list[PredictionRecord]) -> dict[str, dict | float | int | str | None]:
    records, duplicate_count = latest_records(records)
    per_task = {}
    for task in ("orh", "tph", "sdh", "efh", "enfh"):
        per_task[task] = _pair_stats([record for record in records if record.task == task])

    per_task["avg"] = {"accuracy": strict_macro_average({
        task: per_task[task]["accuracy"] for task in ("orh", "tph", "sdh", "efh", "enfh")
    })}
    per_task["metric"] = "strict_pair_accuracy"
    per_task["n_duplicate_records_ignored"] = duplicate_count
    return per_task
