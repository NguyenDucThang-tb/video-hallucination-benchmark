from __future__ import annotations

from src.benchmarks.base import BenchmarkSample
from src.evaluation.parsers import (
    ParseResult,
    parse_ab_ba,
    parse_leading_yes_no,
    parse_mcq,
    parse_yes_no,
    parse_vidhalluc_sth,
    parse_vidhalluc_tsh_official,
)
from src.benchmarks.tempcompass.parsers import (
    parse_caption_matching,
    parse_multi_choice,
    parse_yes_no as parse_tempcompass_yes_no,
)


def normalize_prediction(sample: BenchmarkSample, raw_output: str) -> ParseResult:
    if sample.answer_type == "tempcompass_multi_choice":
        return parse_multi_choice(raw_output, sample.metadata["official_answer"])
    if sample.answer_type == "tempcompass_yes_no":
        return parse_tempcompass_yes_no(raw_output)
    if sample.answer_type == "tempcompass_caption_matching":
        return parse_caption_matching(raw_output, sample.metadata["caption_options"])
    if sample.answer_type == "tempcompass_captioning":
        return ParseResult(raw_output.replace("</s>", "").strip(), "valid")
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
    if sample.benchmark == "vidhalluc" and sample.task == "sth":
        parsed, _ = parse_vidhalluc_sth(raw_output)
        return parsed
    return ParseResult(raw_output.strip(), "valid")
