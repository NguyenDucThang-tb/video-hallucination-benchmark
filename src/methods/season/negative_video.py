from __future__ import annotations

import numpy as np


def spatial_negative(frames: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    """VCD-style Gaussian pixel negative, deterministic for a sample seed."""
    rng = np.random.default_rng(seed)
    values = np.asarray(frames)
    scale = 255.0 if np.issubdtype(values.dtype, np.integer) else 1.0
    noisy = values.astype(np.float32) + rng.normal(0.0, noise_std * scale, values.shape)
    return np.clip(noisy, 0.0, scale).astype(values.dtype)


def temporal_homogenize(pre_layer_features: np.ndarray, beta: float) -> np.ndarray:
    """h[l,t]=(1-beta)h'[l,t]+beta*mean_t(h'[l,t]).

    This operation must be called after every vision layer, not merely once on
    final embeddings. The model adapter owns that recurrent hook.
    """
    features = np.asarray(pre_layer_features)
    if features.ndim < 2:
        raise ValueError("Expected a frame axis followed by feature axes")
    context = features.mean(axis=0, keepdims=True)
    return (1.0 - beta) * features + beta * context
