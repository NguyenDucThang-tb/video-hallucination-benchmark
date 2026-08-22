import numpy as np
import pytest

from src.methods.tcd import TCDConfig, chronological_downsample, contrast_logits
from src.models.base import select_decode_input_ids
from src.models.qwen25_vl import Qwen25VLAdapter, cached_mrope_position_ids


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


def test_cached_decode_forwards_only_the_newest_token():
    input_ids = np.arange(12).reshape(1, 12)
    token_types = np.zeros((1, 12), dtype=np.int64)
    assert select_decode_input_ids(input_ids, None).shape == (1, 12)
    np.testing.assert_array_equal(
        select_decode_input_ids(input_ids, object()),
        input_ids[:, -1:],
    )
    assert select_decode_input_ids(token_types, object()).shape == (1, 1)


def test_cached_qwen_mrope_positions_cover_only_newest_token():
    attention_mask = np.ones((2, 12), dtype=np.int64)
    rope_deltas = np.asarray([[100], [200]], dtype=np.int64)

    position_ids = cached_mrope_position_ids(attention_mask, rope_deltas)

    assert position_ids.shape == (3, 2, 1)
    np.testing.assert_array_equal(position_ids[:, 0, 0], [111, 111, 111])
    np.testing.assert_array_equal(position_ids[:, 1, 0], [211, 211, 211])
