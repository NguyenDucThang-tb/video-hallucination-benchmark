import pytest

from src.models.compatibility import check_compatibility


def test_only_gpu_validated_tcd_adapter_is_enabled():
    ready, reason = check_compatibility("qwen2.5-vl-7b", "tcd")
    assert ready is True
    assert reason

    for model in ("llava-ov-7b", "llava-video-7b"):
        ready, reason = check_compatibility(model, "tcd")
        assert ready is False
        assert reason


def test_unknown_pair_fails():
    with pytest.raises(ValueError):
        check_compatibility("made-up", "base")
