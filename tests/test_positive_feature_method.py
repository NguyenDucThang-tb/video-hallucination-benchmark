import numpy as np
import pytest

from scripts.run_benchmark import instantiate_method
from src.methods.positive_feature import PositiveFeatureMethod
from src.methods.positive_feature.enhancement import enhance_tensor_by_frame_saliency
from src.methods.season.positive_features import FeatureEnhancementConfig
from src.models.base import GenerationConfig
from src.models.llava_ov import LlavaOVAdapter
from src.models.llava_video import LlavaVideoAdapter
from src.models.compatibility import check_compatibility
from src.models.qwen25_vl import Qwen25VLAdapter


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


@pytest.mark.parametrize("adapter_class", [Qwen25VLAdapter, LlavaOVAdapter, LlavaVideoAdapter])
def test_supported_adapters_expose_positive_feature_contract(adapter_class):
    adapter = object.__new__(adapter_class)
    assert adapter.supports_positive_feature_hooks is True
    assert callable(adapter.generate_positive_feature)


@pytest.mark.parametrize("model_name", ["qwen2.5-vl-7b", "llava-ov-7b", "llava-video-7b"])
def test_positive_feature_adapters_remain_gated_until_gpu_validation(model_name):
    ready, reason = check_compatibility(model_name, "positive_feature")
    assert ready is False
    assert "NOT VALIDATED" in reason


def test_shared_enhancement_modifies_frame_grouped_tokens():
    torch = pytest.importorskip("torch")
    tensor = torch.arange(24, dtype=torch.float32).reshape(8, 3) + 1.0
    enhanced, applied, diagnostics = enhance_tensor_by_frame_saliency(
        tensor,
        np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
        FeatureEnhancementConfig(),
        torch,
    )

    assert applied is True
    assert enhanced.shape == tensor.shape
    assert not torch.allclose(enhanced, tensor)
    assert diagnostics["positive_feature_frame_count"] == 4
    assert diagnostics["positive_feature_tokens_per_frame"] == 2


def test_shared_enhancement_supports_frame_major_projector_output():
    torch = pytest.importorskip("torch")
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3) + 1.0
    enhanced, applied, diagnostics = enhance_tensor_by_frame_saliency(
        tensor,
        np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
        FeatureEnhancementConfig(),
        torch,
    )

    assert applied is True
    assert enhanced.shape == tensor.shape
    assert not torch.allclose(enhanced, tensor)
    assert diagnostics["positive_feature_tensor_layout"] == "frame_major"
