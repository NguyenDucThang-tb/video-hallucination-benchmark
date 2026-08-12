import numpy as np

from src.methods.season import diagnostic_weights, season_logits, temporal_homogenize


def test_temporal_homogenization_preserves_shape_and_reduces_variation():
    features = np.array([[0.0, 2.0], [4.0, 6.0]])
    mixed = temporal_homogenize(features, 0.5)
    assert mixed.shape == features.shape
    assert mixed.var(axis=0).mean() < features.var(axis=0).mean()


def test_diagnostic_weights_sum_to_one():
    values = diagnostic_weights(
        np.array([0.8, 0.2]), np.array([0.2, 0.8]), np.array([0.7, 0.3])
    )
    assert np.isclose(values[0] + values[1], 1.0)
    assert values[0] > values[1]


def test_season_logits_formula():
    result = season_logits(
        np.array([2.0]), np.array([1.0]), np.array([0.0]), 1.0, 0.25, 0.75
    )
    assert np.allclose(result, [3.75])
