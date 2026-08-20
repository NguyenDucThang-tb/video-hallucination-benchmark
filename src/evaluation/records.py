from __future__ import annotations

from src.data.schema import PredictionRecord


def latest_records(records: list[PredictionRecord]) -> tuple[list[PredictionRecord], int]:
    """Return the latest record for each benchmark identity and duplicate count."""
    latest: dict[tuple[str, str, str, str, str], PredictionRecord] = {}
    for record in records:
        key = (
            record.sample_id,
            record.model,
            record.method,
            record.benchmark,
            record.task,
        )
        latest[key] = record
    return list(latest.values()), len(records) - len(latest)
