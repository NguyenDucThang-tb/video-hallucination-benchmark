from __future__ import annotations

import re

from src.data.schema import PredictionRecord
from src.evaluation.metrics import accuracy, macro_average


def _parse_sth_scene_change(text: str) -> str | None:
    match = re.search(r"scene\s*change\s*:\s*(yes|no)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    fallback = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    fallback = list(dict.fromkeys(x.lower() for x in fallback))
    if len(fallback) == 1:
        return fallback[0]
    return None


def evaluate_classification(records: list[PredictionRecord]) -> dict:
    tasks = {}
    for task in ("bqa", "mcq", "sth", "tsh"):
        if task == "bqa":
            score, count, correct = accuracy(r.is_correct for r in records if r.task == task)
            tasks[task] = {"accuracy": score, "n": count, "correct": correct}
            continue
        if task == "mcq":
            score, count, correct = accuracy(r.is_correct for r in records if r.task in {"ach", "mcq"})
            tasks[task] = {"accuracy": score, "n": count, "correct": correct}
            continue
        if task == "sth":
            correct_flags = []
            for record in records:
                if record.task != task:
                    continue
                parsed = _parse_sth_scene_change(record.raw_output)
                gt = str(record.metadata.get("scene_change", "")).lower()
                correct_flags.append(None if parsed is None else parsed == gt)
            score, count, correct = accuracy(correct_flags)
            tasks[task] = {"accuracy": score, "n": count, "correct": correct}
            continue
        score, count, correct = accuracy(r.is_correct for r in records if r.task == task)
        tasks[task] = {"accuracy": score, "n": count, "correct": correct}
    tasks["avg"] = {
        "accuracy": macro_average({task: values["accuracy"] for task, values in tasks.items()})
    }
    return tasks
