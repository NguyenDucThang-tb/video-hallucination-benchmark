import numpy as np
import pytest

from scripts.run_benchmark import instantiate_method
from src.methods.positive_feature import PositiveFeatureMethod
from src.models.base import GenerationConfig


class FakePositiveFeatureAdapter:
    name = "fake-qwen"

    def __init__(self, hook_applied: bool = True):
        self.hook_applied = hook_applied
        self.calls = []

    def generate_positive_feature(self, video_frames, prompt, generation_config, config):
        self.calls.append((video_frames, prompt, generation_config, config))
        return "enhanced answer", {"positive_feature_hook_applied": self.hook_applied}


def test_positive_feature_method_delegates_to_adapter_hook():
    model = FakePositiveFeatureAdapter()
    method = PositiveFeatureMethod(model, {"alpha": 0.4})
    output = method.generate(np.zeros((2, 2, 2, 3), dtype=np.uint8), "prompt", GenerationConfig())

    assert output.text == "enhanced answer"
    assert output.diagnostics["positive_feature_hook_applied"] is True
    assert model.calls[0][3]["alpha"] == 0.4


def test_positive_feature_method_refuses_silent_base_fallback():
    method = PositiveFeatureMethod(FakePositiveFeatureAdapter(hook_applied=False), {})
    with pytest.raises(RuntimeError, match="refusing"):
        method.generate(np.zeros((2, 2, 2, 3), dtype=np.uint8), "prompt", GenerationConfig())


def test_runner_registers_positive_feature_method():
    method = instantiate_method("positive_feature", FakePositiveFeatureAdapter())
    assert isinstance(method, PositiveFeatureMethod)
