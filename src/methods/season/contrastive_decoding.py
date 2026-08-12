from __future__ import annotations

import numpy as np


def season_logits(original: np.ndarray, spatial: np.ndarray, temporal: np.ndarray, alpha: float, w_spatial: float, w_temporal: float) -> np.ndarray:
    original = np.asarray(original, dtype=np.float64)
    spatial = np.asarray(spatial, dtype=np.float64)
    temporal = np.asarray(temporal, dtype=np.float64)
    if original.shape != spatial.shape or original.shape != temporal.shape:
        raise ValueError("SEASON branch logits must share shape")
    return (1.0 + alpha) * original - alpha * (w_spatial * spatial + w_temporal * temporal)
