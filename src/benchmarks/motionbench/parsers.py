from __future__ import annotations

from src.evaluation.parsers import ParseResult


def polish_answer(text: str) -> str:
    """Port MotionBench's upstream ``polish_answer`` normalization."""
    answer = text.strip().split(")", 1)[0].strip()
    if "(" in answer:
        answer = answer.split("(", 1)[1].strip()
    answer = answer.split(" ", 1)[0]
    return answer[0].upper() if answer else ""


def parse_motionbench_mcq(text: str) -> ParseResult:
    value = polish_answer(text)
    if value in {"A", "B", "C", "D"}:
        return ParseResult(value, "valid")
    return ParseResult(None, "unparseable", "official polish_answer did not produce A-D")
