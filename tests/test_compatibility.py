import pytest

from src.models.compatibility import check_compatibility


def test_unvalidated_combinations_are_not_claimed_supported():
    for model in ("llava-ov-7b", "qwen2.5-vl-7b", "llava-video-7b"):
        for method in ("base", "tcd", "dino_heal", "season"):
            ready, reason = check_compatibility(model, method)
            assert ready is False
            assert reason


def test_unknown_pair_fails():
    with pytest.raises(ValueError):
        check_compatibility("made-up", "base")
