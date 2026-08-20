from __future__ import annotations

from collections.abc import Iterable, Mapping


def macro_average(values: Mapping[str, float | None]) -> float | None:
    valid = [float(value) for value in values.values() if value is not None]
    return sum(valid) / len(valid) if valid else None


def strict_macro_average(values: Mapping[str, float | None]) -> float | None:
    """Average a fixed metric family only when every component is available."""
    if not values or any(value is None for value in values.values()):
        return None
    return sum(float(value) for value in values.values()) / len(values)


def accuracy(correct: Iterable[bool | None]) -> tuple[float | None, int, int]:
    values = list(correct)
    total = len(values)
    correct_count = sum(value is True for value in values)
    return (correct_count / total if total else None, total, correct_count)
