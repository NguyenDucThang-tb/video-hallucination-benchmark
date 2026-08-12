from __future__ import annotations

from collections.abc import Iterable, Mapping


def macro_average(values: Mapping[str, float | None]) -> float | None:
    valid = [float(value) for value in values.values() if value is not None]
    return sum(valid) / len(valid) if valid else None


def accuracy(correct: Iterable[bool | None]) -> tuple[float | None, int, int]:
    valid = [value for value in correct if value is not None]
    return ((sum(valid) / len(valid)) if valid else None, len(valid), sum(valid))
