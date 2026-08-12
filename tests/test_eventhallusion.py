import pytest

from src.benchmarks.eventhallusion.evaluator import description_reference


def test_mix_uses_unexpected_event_not_caption():
    info = {"caption": "ordinary event", "unexpected": "unexpected event"}
    assert description_reference("mix", info) == "unexpected event"
    assert description_reference("entire", info) == "ordinary event"


def test_unknown_split_fails():
    with pytest.raises(ValueError):
        description_reference("interleave", {})
