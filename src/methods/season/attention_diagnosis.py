from __future__ import annotations

import numpy as np


def _probabilities(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Frame attention scores must be a non-empty vector")
    if not np.all(np.isfinite(x)):
        raise ValueError("Frame attention scores contain non-finite values")
    x = x - np.max(x)
    exp = np.exp(x)
    denominator = float(exp.sum())
    if not np.isfinite(denominator) or denominator <= epsilon:
        raise ValueError("Frame attention softmax has an invalid denominator")
    return exp / denominator


def frame_attention(attention: np.ndarray) -> np.ndarray:
    """Aggregate selected layer/head attention [L,H,T,P] to softmax over T."""
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("Expected attention [layers, heads, frames, patches]")
    per_frame = values.sum(axis=(0, 1, 3))
    return _probabilities(per_frame)


def jensen_shannon(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-8) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1 or p.size == 0:
        raise ValueError("JSD inputs must be non-empty vectors with the same shape")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("JSD inputs contain non-finite values")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("JSD inputs must be non-negative")
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= epsilon or q_sum <= epsilon:
        raise ValueError("JSD inputs must have positive mass")
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
    kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))
    result = float(0.5 * (kl_pm + kl_qm))
    if not np.isfinite(result):
        raise ValueError("JSD produced a non-finite result")
    return max(0.0, result)


def diagnostic_weights(original: np.ndarray, spatial: np.ndarray, temporal: np.ndarray, epsilon: float = 1e-8) -> tuple[float, float, float, float]:
    d_spatial = jensen_shannon(original, spatial, epsilon)
    d_temporal = jensen_shannon(original, temporal, epsilon)
    denominator = d_spatial + d_temporal
    if denominator <= epsilon:
        return 0.5, 0.5, d_spatial, d_temporal
    weights = d_spatial / denominator, d_temporal / denominator
    if not np.all(np.isfinite(weights)):
        raise ValueError("SEASON diagnostic weights are non-finite")
    return weights[0], weights[1], d_spatial, d_temporal
