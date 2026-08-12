from __future__ import annotations

from src.data.schema import PredictionRecord
from src.evaluation.metrics import accuracy, macro_average


def evaluate_classification(records: list[PredictionRecord]) -> dict:
    tasks = {}
    for task in ("bqa", "mcq", "sth", "tsh"):
        score, count, correct = accuracy(r.is_correct for r in records if r.task == task)
        tasks[task] = {"accuracy": score, "n": count, "correct": correct}
    tasks["avg"] = {
        "accuracy": macro_average({task: values["accuracy"] for task, values in tasks.items()})
    }
    return tasks
