from __future__ import annotations

import numpy as np


def _is_torch_tensor(value) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "isfinite")


def season_logits(original, spatial, temporal, alpha: float, w_spatial: float, w_temporal: float):
    if original.shape != spatial.shape or original.shape != temporal.shape:
        raise ValueError("SEASON branch logits must share shape")
    if original.ndim != 1 or original.shape[0] == 0:
        raise ValueError("SEASON branch logits must be non-empty vocabulary vectors")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("SEASON alpha must be finite and non-negative")
    if not np.isfinite(w_spatial) or not np.isfinite(w_temporal):
        raise ValueError("SEASON diagnostic weights must be finite")
    if abs((w_spatial + w_temporal) - 1.0) > 1e-5:
        raise ValueError("SEASON diagnostic weights must sum to one")

    if _is_torch_tensor(original):
        import torch

        if not all(_is_torch_tensor(value) for value in (spatial, temporal)):
            raise TypeError("SEASON branch logits must use the same tensor backend")
        if not all(bool(torch.isfinite(value).all()) for value in (original, spatial, temporal)):
            raise ValueError("SEASON branch logits contain non-finite values")
        result = (1.0 + alpha) * original - alpha * (
            w_spatial * spatial + w_temporal * temporal
        )
        if not bool(torch.isfinite(result).all()):
            raise ValueError("SEASON combined logits contain non-finite values")
        return result

    original = np.asarray(original, dtype=np.float64)
    spatial = np.asarray(spatial, dtype=np.float64)
    temporal = np.asarray(temporal, dtype=np.float64)
    if not all(np.all(np.isfinite(value)) for value in (original, spatial, temporal)):
        raise ValueError("SEASON branch logits contain non-finite values")
    result = (1.0 + alpha) * original - alpha * (
        w_spatial * spatial + w_temporal * temporal
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("SEASON combined logits contain non-finite values")
    return result
