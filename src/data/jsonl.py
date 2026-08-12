from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import PredictionRecord


def read_jsonl(path: str | Path) -> list[PredictionRecord]:
    source = Path(path)
    if not source.exists():
        return []
    records = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(PredictionRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
    return records


def append_jsonl(path: str | Path, record: PredictionRecord) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_jsonl(path: str | Path, records: Iterable[PredictionRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def valid_resume_keys(path: str | Path) -> set[tuple[str, str, str, str, str]]:
    return {
        (r.sample_id, r.model, r.method, r.benchmark, r.task)
        for r in read_jsonl(path)
        if r.is_valid_for_resume
    }
