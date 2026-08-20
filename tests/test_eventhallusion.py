from pathlib import Path

import pytest

from src.benchmarks.eventhallusion.evaluator import description_reference
from src.benchmarks.eventhallusion.evaluator import evaluate_binary
from src.benchmarks.eventhallusion.loader import normalize_yes_no_label, resolve_video_path
from src.data.schema import PredictionRecord
from src.evaluation.parsers import parse_leading_yes_no


def test_mix_uses_unexpected_event_not_caption():
    info = {"caption": "ordinary event", "unexpected": "unexpected event"}
    assert description_reference("mix", info) == "unexpected event"
    assert description_reference("interleave", info) == "unexpected event"
    assert description_reference("entire", info) == "ordinary event"


def test_unknown_split_fails():
    with pytest.raises(ValueError):
        description_reference("unknown", {})


def test_mix_falls_back_to_interleave_video_directory(tmp_path: Path):
    video = tmp_path / "interleave" / "123.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    assert resolve_video_path(tmp_path, "mix", "123") == video


def test_ground_truth_yes_no_labels_drop_trailing_punctuation():
    assert normalize_yes_no_label("No.") == "no"
    assert normalize_yes_no_label("Yes") == "yes"


def test_official_parser_requires_answer_at_start():
    assert parse_leading_yes_no("Yes, it does.").value == "yes"
    assert parse_leading_yes_no("I think yes.").value is None


def test_unparseable_event_answer_remains_in_denominator():
    records = [
        PredictionRecord(
            sample_id=str(index), model="m", method="base", benchmark="eventhallusion",
            task="entire", prompt="q", frame_indices=[], raw_output=raw,
            normalized_output=value, ground_truth="yes", is_correct=correct,
            parser_status=status,
        )
        for index, (raw, value, correct, status) in enumerate([
            ("Yes", "yes", True, "valid"),
            ("Maybe", None, None, "unparseable"),
        ])
    ]
    result = evaluate_binary(records)
    assert result["entire"]["n"] == 2
    assert result["entire"]["accuracy"] == 0.5
