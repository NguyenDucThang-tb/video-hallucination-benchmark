import pytest

from scripts.compare_vidhalluc_evaluators import upstream_sth_parse
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
