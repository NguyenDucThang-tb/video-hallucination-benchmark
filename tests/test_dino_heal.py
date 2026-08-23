import numpy as np
import pytest

from src.methods.dino_heal.dino_heal_method import DINOHealMethod
from src.methods.dino_heal.fusion import DINOHealConfig, fuse_saliency
from src.models.base import GenerationConfig


def test_patch_fusion_shape_and_weights():
    features = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    saliency = np.linspace(0, 1, 6, dtype=np.float32).reshape(2, 3)
    config = DINOHealConfig(visual_weight=0.3, saliency_weight=0.7)
    result = fuse_saliency(features, saliency, config)
    normalized = (features - features.mean()) / (features.std() + 1e-6)
    assert result.shape == features.shape
    assert np.allclose(result, 0.3 * normalized + 0.7 * saliency[..., None])


def test_patch_fusion_rejects_frame_only_saliency():
    with pytest.raises(ValueError):
        fuse_saliency(np.zeros((2, 3, 4)), np.zeros((2,)), DINOHealConfig())


def test_qwen_frame_saliency_can_be_represented_as_single_patch():
    features = np.zeros((8, 1, 1), dtype=np.float32)
    frame_saliency = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    result = fuse_saliency(features, frame_saliency[:, None], DINOHealConfig())
    assert result.shape == features.shape
    assert result[-1, 0, 0] > result[0, 0, 0]


def test_dino_failure_has_no_silent_base_fallback():
    class FailedDinoModel:
        name = "failed"

        def generate_dino_heal(self, *args, **kwargs):
            return "base-looking answer", {"dino_loaded": False}

    method = DINOHealMethod(FailedDinoModel(), {})
    with pytest.raises(RuntimeError, match="refusing"):
        method.generate(np.zeros((8, 2, 2, 3)), "q", GenerationConfig())
