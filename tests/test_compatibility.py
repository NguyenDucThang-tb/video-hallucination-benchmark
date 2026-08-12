import pytest

from src.models.compatibility import check_compatibility


def test_unvalidated_combinations_are_not_claimed_supported():
    ready, reason = check_compatibility("qwen2.5-vl-7b", "dino_heal")
    assert ready is False
    assert "Qwen" in reason


def test_unknown_pair_fails():
    with pytest.raises(ValueError):
        check_compatibility("made-up", "base")
