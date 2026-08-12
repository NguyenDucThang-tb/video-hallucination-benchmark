from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DINOHealConfig:
    visual_weight: float = 0.3
    saliency_weight: float = 0.7
    require_dino: bool = True


def fuse_saliency(features: np.ndarray, saliency: np.ndarray, config: DINOHealConfig) -> np.ndarray:
    """Reproduce VidHalluc's feature/saliency fusion without silent fallback."""
    values = np.asarray(features, dtype=np.float32)
    scores = np.asarray(saliency, dtype=np.float32)
    if values.ndim != 3 or scores.shape != values.shape[:2]:
        raise ValueError("Expected features [frames,patches,dim], saliency [frames,patches]")
    normalized = (values - values.mean()) / (values.std() + 1e-6)
    return config.visual_weight * normalized + config.saliency_weight * scores[..., None]
