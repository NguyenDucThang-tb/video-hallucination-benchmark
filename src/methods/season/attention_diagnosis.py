from __future__ import annotations

import numpy as np


def _probabilities(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / max(float(exp.sum()), epsilon)


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
    p = p / max(float(p.sum()), epsilon)
    q = q / max(float(q.sum()), epsilon)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
    kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))
    return float(0.5 * (kl_pm + kl_qm))


def diagnostic_weights(original: np.ndarray, spatial: np.ndarray, temporal: np.ndarray, epsilon: float = 1e-8) -> tuple[float, float, float, float]:
    d_spatial = jensen_shannon(original, spatial, epsilon)
    d_temporal = jensen_shannon(original, temporal, epsilon)
    denominator = d_spatial + d_temporal
    if denominator <= epsilon:
        return 0.5, 0.5, d_spatial, d_temporal
    return d_spatial / denominator, d_temporal / denominator, d_spatial, d_temporal
