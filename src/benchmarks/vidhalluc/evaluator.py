from __future__ import annotations

import math
from collections import defaultdict

from src.data.schema import PredictionRecord
from src.evaluation.metrics import accuracy, strict_macro_average
from src.evaluation.parsers import parse_vidhalluc_sth, parse_vidhalluc_tsh_official
from src.evaluation.records import latest_records


def _official_bqa_answer(text: str) -> str:
    return text.strip().split(".", 1)[0].split(",", 1)[0].strip().upper()


def _binary_mcc(ground_truth: list[bool], predictions: list[bool]) -> float:
    tp = sum(gt and pred for gt, pred in zip(ground_truth, predictions))
    tn = sum((not gt) and (not pred) for gt, pred in zip(ground_truth, predictions))
    fp = sum((not gt) and pred for gt, pred in zip(ground_truth, predictions))
    fn = sum(gt and (not pred) for gt, pred in zip(ground_truth, predictions))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / denominator) if denominator else 0.0


def _bqa_metric(records: list[PredictionRecord]) -> dict:
    questions: dict[tuple[str, int], list[PredictionRecord]] = defaultdict(list)
    for record in records:
        if record.task != "bqa":
            continue
        key = (str(record.metadata.get("section", "")), int(record.metadata.get("question_index", -1)))
        questions[key].append(record)
    correct = 0
    incomplete = 0
    for items in questions.values():
        expected = max(int(item.metadata.get("expected_clip_count", len(items))) for item in items)
        hits = [
            _official_bqa_answer(item.raw_output) == str(item.ground_truth).strip().upper()
            for item in items
        ]
        complete = len(items) == expected
        incomplete += int(not complete)
        correct += int(complete and all(hits))
    total = len(questions)
    return {
        "accuracy": correct / total if total else None,
        "n": total,
        "correct": correct,
        "n_incomplete_questions": incomplete,
        "unit": "question_all_clips_correct",
    }


def _sth_metric(records: list[PredictionRecord]) -> dict:
    items = [record for record in records if record.task == "sth"]
    ground_truth = [str(item.metadata.get("scene_change", item.ground_truth)).lower() == "yes" for item in items]
    parsed_with_locations = [parse_vidhalluc_sth(item.raw_output) for item in items]
    parsed = [result.value for result, _ in parsed_with_locations]
    predictions = [value == "yes" for value in parsed]
    mcc = _binary_mcc(ground_truth, predictions) if items else None
    classification = ((mcc + 1.0) / 2.0) ** 2 if mcc is not None else None
    binary_correct = sum(value is not None and (value == ("yes" if gt else "no")) for value, gt in zip(parsed, ground_truth))
    return {
        "accuracy": None,
        "official_accuracy": None,
        "official_status": "SIMCSE_NOT_AVAILABLE",
        "status": "N/A",
        "reason": "Official STH requires SimCSE location-description scoring; it was not executed.",
        "n": len(items),
        "n_parser_error": sum(value is None for value in parsed),
        "binary_accuracy_diagnostic": binary_correct / len(items) if items else None,
        "mcc": mcc,
        "classification_score": classification,
        "description_accuracy": None,
        "overall_score": None,
        "invalid_prediction_policy_for_mcc": "upstream-compatible: values other than yes are treated as no",
        "formula": "0.6 * classification_score + 0.4 * description_accuracy",
    }


def _tsh_metric(records: list[PredictionRecord]) -> dict:
    items = [record for record in records if record.task == "tsh"]
    parses = [parse_vidhalluc_tsh_official(item.raw_output) for item in items]
    valid = [parse.value is not None for parse in parses]
    correct = [
        parse.value is not None and parse.value == str(item.ground_truth).strip().upper()
        for item, parse in zip(items, parses)
    ]
    total = len(items)
    valid_count = sum(valid)
    correct_count = sum(correct)
    runtime_failures = sum(bool(item.error) for item in items)
    return {
        "accuracy": correct_count / total if total else None,
        "official_accuracy": correct_count / total if total else None,
        "all_sample_accuracy": correct_count / total if total else None,
        "valid_only_accuracy": correct_count / valid_count if valid_count else None,
        "parse_coverage": valid_count / total if total else None,
        "n": total,
        "correct": correct_count,
        "valid_count": valid_count,
        "valid_incorrect": valid_count - correct_count,
        "unparseable_count": total - valid_count,
        "runtime_failure_count": runtime_failures,
        "denominator_policy": "official VidHalluc: every annotation is in the denominator",
        "parser": "CyL97/VidHalluc eval/evaluation/eval_tsh.py",
    }


def evaluate_classification(records: list[PredictionRecord]) -> dict:
    records, duplicate_count = latest_records(records)
    tasks = {}
    for task in ("bqa", "mcq", "sth", "tsh"):
        if task == "bqa":
            tasks[task] = _bqa_metric(records)
            continue
        if task == "mcq":
            score, count, correct = accuracy(r.is_correct for r in records if r.task in {"ach", "mcq"})
            tasks[task] = {"accuracy": score, "n": count, "correct": correct}
            continue
        if task == "sth":
            tasks[task] = _sth_metric(records)
            continue
        if task == "tsh":
            tasks[task] = _tsh_metric(records)
            continue
        score, count, correct = accuracy(r.is_correct for r in records if r.task == task)
        tasks[task] = {"accuracy": score, "n": count, "correct": correct}
    tasks["avg"] = {"accuracy": strict_macro_average({
        task: tasks[task]["accuracy"] for task in ("bqa", "mcq", "sth", "tsh")
    }), "formula": "macro average of BQA, MCQ, official STH, and TSH; N/A if any task is unavailable"}
    tasks["n_duplicate_records_ignored"] = duplicate_count
    return tasks
