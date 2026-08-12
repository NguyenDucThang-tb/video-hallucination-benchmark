import numpy as np

from src.methods.tcd import chronological_downsample, contrast_logits


def test_tcd_negative_is_chronological_subset_of_original_eight():
    frames = np.arange(8)[:, None]
    negative, positions = chronological_downsample(frames, 4)
    assert positions == sorted(positions)
    assert positions == [0, 2, 5, 7]
    assert negative[:, 0].tolist() == positions


def test_tcd_contrast_and_mask():
    result = contrast_logits(np.array([2.0, 1.0]), np.array([1.0, 2.0]), 0.5, 0.5)
    assert result[0] == 2.5
    assert np.isneginf(result[1])
