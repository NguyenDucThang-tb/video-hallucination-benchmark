import numpy as np
import pytest

from src.methods.dino_heal.fusion import DINOHealConfig, fuse_saliency


def test_dino_fusion_requires_one_saliency_value_per_visual_patch():
    features = np.zeros((8, 16, 32), dtype=np.float32)
    patch_saliency = np.zeros((8, 16), dtype=np.float32)
    assert fuse_saliency(features, patch_saliency, DINOHealConfig()).shape == features.shape
    with pytest.raises(ValueError):
        fuse_saliency(features, np.zeros((8, 1), dtype=np.float32), DINOHealConfig())
