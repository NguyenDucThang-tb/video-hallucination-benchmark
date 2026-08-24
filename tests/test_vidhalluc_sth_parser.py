import pytest

from scripts.compare_vidhalluc_evaluators import upstream_sth_parse
from src.benchmarks.base import BenchmarkSample
from src.evaluation.normalize import normalize_prediction
from src.evaluation.parsers import parse_vidhalluc_sth


@pytest.mark.parametrize(("text", "expected", "locations"), [
    ("Scene change: Yes, Locations: from a kitchen to a living room.", "yes", "from a kitchen to a living room."),
    ("Scene change: No, Locations: None", "no", "None"),
    ("Scene change: Yes; Locations: from street to river", None, None),
    ("There is a scene change from a street to a river", None, None),
    ("No scene transition detected", None, None),
    ("I am not sure", None, None),
    ("", None, None),
])
def test_sth_parser_keeps_unknown_separate(text, expected, locations):
    parsed, parsed_locations = parse_vidhalluc_sth(text)
    assert parsed.value == expected
    assert parsed_locations == locations


def test_upstream_sth_non_yes_behavior_is_explicit():
    official, _ = upstream_sth_parse("I am not sure")
    local, _ = parse_vidhalluc_sth("I am not sure")
    assert official == "I am not sure"
    assert local.value is None


def test_sth_record_normalization_uses_scene_change_field(tmp_path):
    sample = BenchmarkSample(
        sample_id="sth:one",
        benchmark="vidhalluc",
        task="sth",
        video_path=tmp_path / "one.mp4",
        prompt="question",
        ground_truth="no",
        answer_type="text",
    )
    parsed = normalize_prediction(sample, "Scene change: No, Locations: None.")
    assert parsed.value == "no"
    assert parsed.status == "valid"
