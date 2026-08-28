import importlib.util
from pathlib import Path

import pytest

from src.evaluation.parsers import parse_vidhalluc_tsh_official


UPSTREAM = Path(__file__).resolve().parents[1] / "external/VidHalluc/eval/evaluation/eval_tsh.py"
SPEC = importlib.util.spec_from_file_location("upstream_vidhalluc_tsh_test", UPSTREAM)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("text", [
    "AB", "BA", "A", "B", "Action A before Action B", "Action B before Action A",
    "Action A happens first, then Action B", "Action B happens before Action A",
    "Only Action A is visible", "Only Action B is visible", "It is not clear",
    "No clear order", "", "AB.",
])
def test_local_tsh_compatibility_parser_matches_vendored_upstream(text):
    upstream = MODULE.model_answer_to_correct_answer(text)
    local = parse_vidhalluc_tsh_official(text).value
    expected = {"A": "AB", "B": "BA"}.get(upstream, upstream)
    assert local == (None if expected == "None" else expected)
