import pytest

from src.evaluation.parsers import parse_ab_ba, parse_vidhalluc_sth, parse_vidhalluc_tsh_official


@pytest.mark.parametrize(("text", "expected"), [
    ("AB", "AB"), ("BA", "BA"), ("A", "A"), ("B", "B"),
    ("Action A happens before Action B", "AB"),
    ("Action B happens before Action A", "BA"),
    ("Action B happens after Action A", "BA"),
    ("A occurs first, followed by B", None),
    ("The first action is A and the second is B", None),
    ("not clear", None), ("no clear", None), ("", None), ("AB.", None),
])
def test_official_tsh_parser_matches_upstream_cases(text, expected):
    assert parse_vidhalluc_tsh_official(text).value == expected


def test_diagnostic_parser_is_explicitly_more_permissive_than_official():
    assert parse_vidhalluc_tsh_official("AB.").value is None
    assert parse_ab_ba("AB.").value == "AB"


def test_sth_parser_preserves_scene_and_location_fields():
    parsed, locations = parse_vidhalluc_sth("Scene change: Yes, Locations: from a kitchen to a street.")
    assert parsed.value == "yes"
    assert locations == "from a kitchen to a street."


def test_sth_unparseable_output_stays_none():
    parsed, _ = parse_vidhalluc_sth("There may be a transition.")
    assert parsed.value is None

