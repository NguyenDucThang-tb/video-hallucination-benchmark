import pytest

from src.models.compatibility import check_compatibility


def test_tcd_is_disabled_until_post_audit_real_model_validation():
    for model in ("llava-ov-7b", "qwen2.5-vl-7b", "llava-video-7b"):
        ready, reason = check_compatibility(model, "tcd")
        assert ready is False
        assert reason


def test_unknown_pair_fails():
    with pytest.raises(ValueError):
        check_compatibility("made-up", "base")
