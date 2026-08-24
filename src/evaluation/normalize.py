from __future__ import annotations

from src.benchmarks.base import BenchmarkSample
from src.evaluation.parsers import (
    ParseResult,
    parse_ab_ba,
    parse_leading_yes_no,
    parse_mcq,
    parse_yes_no,
    parse_vidhalluc_tsh_official,
)


def normalize_prediction(sample: BenchmarkSample, raw_output: str) -> ParseResult:
    if sample.answer_type == "yes_no":
        if sample.benchmark == "eventhallusion":
            return parse_leading_yes_no(raw_output)
        return parse_yes_no(raw_output)
    if sample.answer_type == "mcq":
        return parse_mcq(raw_output, sample.choices)
    if sample.answer_type == "ab_ba":
        if sample.benchmark == "vidhalluc" and sample.task == "tsh":
            return parse_vidhalluc_tsh_official(raw_output)
        return parse_ab_ba(raw_output)
    return ParseResult(raw_output.strip(), "valid")
