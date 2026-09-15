from __future__ import annotations

from src.evaluation.parsers import ParseResult


def _clean(text: str) -> str:
    return text.replace("</s>", "").strip()


def parse_multi_choice(text: str, official_answer: str) -> ParseResult:
    """Port the hand-crafted matching order from eval_multi-choice.py."""
    prediction = _clean(text)
    if prediction == official_answer:
        return ParseResult(official_answer[0], "valid")
    if prediction in {"A", "B", "C", "D"}:
        return ParseResult(prediction, "valid")
    if prediction.startswith(("A.", "B.", "C.", "D.")):
        return ParseResult(prediction.split(".", 1)[0], "valid")
    if prediction.startswith(("A)", "B)", "C)", "D)")):
        return ParseResult(prediction.split(")", 1)[0], "valid")
    return ParseResult(None, "requires_judge")


def parse_yes_no(text: str) -> ParseResult:
    """Port TempCompass eval_yes_no.extract_pred exactly."""
    prediction = _clean(text).lower()
    if prediction.startswith("yes"):
        return ParseResult("yes", "valid")
    if prediction.startswith("no"):
        return ParseResult("no", "valid")
    return ParseResult(None, "requires_judge")


def parse_caption_matching(text: str, options: list[dict]) -> ParseResult:
    """Port TempCompass eval_caption_matching.eval_rule matching semantics."""
    prediction = _clean(text)
    for option in options:
        label = option["label"]
        short = option["short"]
        sentence = option["sentence"]
        full = f"{label}: {sentence}"
        sentence_after_paren = prediction.split(") ", 1)[1] if ") " in prediction else None
        if (
            prediction == full
            or prediction == sentence
            or sentence_after_paren == sentence
            or prediction == label
            or prediction.replace(".", "") == label
            or prediction == short
            or prediction.replace(".", "") == short
        ):
            return ParseResult(label, "valid")
    return ParseResult(None, "requires_judge")
