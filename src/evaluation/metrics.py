from __future__ import annotations

from collections.abc import Iterable, Mapping


def macro_average(values: Mapping[str, float | None]) -> float | None:
    valid = [float(value) for value in values.values() if value is not None]
    return sum(valid) / len(valid) if valid else None


def accuracy(correct: Iterable[bool | None]) -> tuple[float | None, int, int]:
    values = list(correct)
    total = len(values)
    correct_count = sum(value is True for value in values)
    return (correct_count / total if total else None, total, correct_count)
