from src.models.compatibility import check_compatibility


def test_approximations_are_not_enabled_as_paper_compatible_methods():
    for model in ("llava-ov-7b", "qwen2.5-vl-7b"):
        for method in ("tcd", "dino_heal", "season"):
            ready, reason = check_compatibility(model, method)
            assert ready is False
            assert reason
