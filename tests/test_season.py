import numpy as np
import pytest

from src.methods.season import (
    blend_temporal_hidden,
    diagnostic_weights,
    frame_mean_context,
    season_logits,
    spatial_negative,
    temporal_homogenize,
)
from src.methods.season import (
    FeatureEnhancementConfig,
    directed_motion_evidence,
    enhance_visual_features,
    foreground_persistence,
)


def test_temporal_homogenization_preserves_shape_and_reduces_variation():
    features = np.array([[0.0, 2.0], [4.0, 6.0]])
    mixed = temporal_homogenize(features, 0.5)
    assert mixed.shape == features.shape
    assert mixed.var(axis=0).mean() < features.var(axis=0).mean()


def test_spatial_negative_is_deterministic_and_changes_pixels():
    frames = np.full((8, 4, 4, 3), 128, dtype=np.uint8)
    first = spatial_negative(frames, noise_std=0.1, seed=7)
    second = spatial_negative(frames, noise_std=0.1, seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, frames)


def test_layer_context_groups_equal_crops_per_frame():
    values = np.arange(8 * 2 * 3, dtype=np.float32).reshape(16, 3)
    context = frame_mean_context(values, frame_count=8)
    assert context.shape == values.shape
    grouped = context.reshape(8, 2, 3)
    assert np.allclose(grouped[0], grouped[-1])


def test_layer_blending_uses_original_context():
    current = np.zeros((8, 2), dtype=np.float32)
    original = np.ones((8, 2), dtype=np.float32) * 4
    blended = blend_temporal_hidden(current, original, beta=0.25)
    assert np.allclose(blended, 1.0)


def test_diagnostic_weights_sum_to_one():
    values = diagnostic_weights(
        np.array([0.8, 0.2]), np.array([0.2, 0.8]), np.array([0.7, 0.3])
    )
    assert np.isclose(values[0] + values[1], 1.0)
    assert values[0] > values[1]


def test_diagnostic_weights_use_equal_fallback_for_identical_attention():
    values = diagnostic_weights(
        np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])
    )
    assert values[:2] == (0.5, 0.5)


def test_diagnostic_weights_reject_nonfinite_attention():
    with pytest.raises(ValueError, match="non-finite"):
        diagnostic_weights(
            np.array([np.nan, 1.0]), np.array([0.5, 0.5]), np.array([0.5, 0.5])
        )


def test_season_logits_formula():
    result = season_logits(
        np.array([2.0]), np.array([1.0]), np.array([0.0]), 1.0, 0.25, 0.75
    )
    assert np.allclose(result, [3.75])


def test_season_logits_reject_shape_and_numerical_errors():
    with pytest.raises(ValueError, match="share shape"):
        season_logits(np.ones(2), np.ones(3), np.ones(2), 1.0, 0.5, 0.5)
    with pytest.raises(ValueError, match="non-finite"):
        season_logits(
            np.array([np.inf, 0.0]), np.ones(2), np.ones(2), 1.0, 0.5, 0.5
        )


def test_foreground_persistence_broadcasts_patch_frequency():
    fg = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    persist = foreground_persistence(fg)
    expected = np.array([[1.0, 2.0 / 3.0, 1.0 / 3.0]] * 3)
    assert persist.shape == fg.shape
    assert np.allclose(persist, expected)


def test_directed_motion_evidence_removes_background_motion():
    features = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[2.0, 0.0], [5.0, 0.0]],
        ],
        dtype=np.float32,
    )
    foreground = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    evidence = directed_motion_evidence(features, foreground)
    assert np.allclose(evidence[0], 0.0)
    assert np.linalg.norm(evidence[1, 0]) > 0.0
    assert np.allclose(evidence[1, 1], 0.0)


def test_enhance_visual_features_applies_spatial_and_temporal_terms():
    features = np.ones((2, 2, 3), dtype=np.float32)
    features[1, 0] = 2.0
    foreground = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    output = enhance_visual_features(
        features,
        foreground,
        FeatureEnhancementConfig(alpha=0.4, alpha_spatial=0.4, beta_temporal=0.4),
    )
    assert output.features.shape == features.shape
    assert output.features.dtype == np.float32
    assert output.spatial_scale[0, 0] > output.spatial_scale[0, 1]
    assert output.features[1, 0].mean() > features[1, 0].mean()
    assert output.diagnostics["mean_relative_delta"] > 0.0
