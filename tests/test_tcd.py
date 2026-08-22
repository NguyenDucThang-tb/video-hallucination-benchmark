import numpy as np
import pytest

from src.methods.tcd import TCDConfig, chronological_downsample, contrast_logits
from src.models.qwen25_vl import Qwen25VLAdapter


def test_tcd_negative_is_chronological_subset_of_original_eight():
    frames = np.arange(8)[:, None]
    negative, positions = chronological_downsample(frames, 4)
    assert positions == sorted(positions)
    assert positions == [0, 2, 5, 7]
    assert len(set(positions)) == 4
    assert negative[:, 0].tolist() == positions


def test_tcd_contrast_and_mask():
    result = contrast_logits(np.array([2.0, 1.0]), np.array([1.0, 2.0]), 0.5, 0.5)
    assert result[0] == 2.5
    assert np.isneginf(result[1])


def test_tcd_threshold_is_applied_to_mixed_raw_logits():
    original = np.array([4.0, 3.0, 1.0])
    negative = np.array([4.0, 0.0, 0.0])
    result = contrast_logits(original, negative, alpha=0.5, beta=0.5)
    np.testing.assert_allclose(result[:2], [4.0, 4.5])
    assert np.isneginf(result[2])


def test_tcd_config_rejects_non_paper_threshold_space():
    with pytest.raises(ValueError, match="raw-logit"):
        TCDConfig(threshold_space="probability")


def test_qwen_tcd_negative_is_not_downsampled_after_processing():
    adapter = Qwen25VLAdapter.__new__(Qwen25VLAdapter)
    processed = {
        "pixel_values": np.zeros((4, 3, 2, 2), dtype=np.float32),
        "image_grid_thw": np.asarray([[1, 2, 2]] * 4),
    }
    transformed = adapter._apply_branch_transform(processed, "tcd_negative")
    assert transformed["pixel_values"].shape[0] == 4
    assert transformed["image_grid_thw"].shape[0] == 4
