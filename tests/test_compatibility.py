import pytest

from src.models.compatibility import check_compatibility


def test_mitigation_methods_are_blocked_until_baseline_gate_passes():
    for model in ("llava-ov-7b", "qwen2.5-vl-7b", "llava-video-7b"):
        ready, reason = check_compatibility(model, "tcd")
        assert ready is False
        assert reason


def test_unknown_pair_fails():
    with pytest.raises(ValueError):
        check_compatibility("made-up", "base")
