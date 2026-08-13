from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureEnhancementConfig:
    """Positive visual-feature enhancement coefficients."""

    alpha: float = 0.4
    alpha_spatial: float = 0.4
    beta_temporal: float = 0.4
    epsilon: float = 1e-6


@dataclass(frozen=True)
class FeatureEnhancementOutput:
    features: np.ndarray
    spatial_scale: np.ndarray
    temporal_evidence: np.ndarray
    diagnostics: dict[str, Any]


def foreground_persistence(foreground: np.ndarray) -> np.ndarray:
    """Return per-patch foreground persistence broadcast over time."""
    fg = np.asarray(foreground, dtype=np.float32)
    if fg.ndim != 2:
        raise ValueError("foreground must have shape [frames, patches]")
    return np.broadcast_to(fg.mean(axis=0, keepdims=True), fg.shape).astype(np.float32)


def directed_motion_evidence(features: np.ndarray, foreground: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Compute background-filtered, norm-stabilized directed motion evidence.

    Evidence follows the meeting formula:

        diff[t,p] = V[t,p] - V[t-1,p]
        e[t,p]    = normalize(diff[t,p]) * ||V[t,p]||

    Frame 0 has no previous frame, so its evidence is zero.
    """
    values = np.asarray(features, dtype=np.float32)
    fg = np.asarray(foreground, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("features must have shape [frames, patches, dim]")
    if fg.shape != values.shape[:2]:
        raise ValueError(f"foreground shape {fg.shape} must match features[:2] {values.shape[:2]}")

    evidence = np.zeros_like(values, dtype=np.float32)
    evidence[1:] = values[1:] - values[:-1]
    evidence *= fg[..., None]

    evidence_norm = np.linalg.norm(evidence, axis=-1, keepdims=True)
    feature_norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return evidence / (evidence_norm + epsilon) * feature_norm


def enhance_visual_features(
    features: np.ndarray,
    foreground: np.ndarray,
    config: FeatureEnhancementConfig | None = None,
) -> FeatureEnhancementOutput:
    """Apply spatial persistence scaling plus directed temporal evidence.

    For visual token V[t,p], foreground saliency F[t,p], and persistence
    P[p] = mean_t F[t,p]:

        S[t,p]  = alpha * F[t,p] + alpha_spatial * P[p]
        V'[t,p] = V[t,p] * (1 + S[t,p]) + beta_temporal * E[t,p]

    Background tokens receive zero temporal evidence because E is multiplied by
    the foreground mask before normalization.
    """
    cfg = config or FeatureEnhancementConfig()
    values = np.asarray(features, dtype=np.float32)
    fg = np.asarray(foreground, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("features must have shape [frames, patches, dim]")
    if fg.shape != values.shape[:2]:
        raise ValueError(f"foreground shape {fg.shape} must match features[:2] {values.shape[:2]}")
    if np.any(fg < 0):
        raise ValueError("foreground saliency must be non-negative")

    persist = foreground_persistence(fg)
    spatial_scale = cfg.alpha * fg + cfg.alpha_spatial * persist
    temporal = directed_motion_evidence(values, fg, cfg.epsilon)
    enhanced = values * (1.0 + spatial_scale[..., None]) + cfg.beta_temporal * temporal

    delta = np.linalg.norm(enhanced - values, axis=-1) / (
        np.linalg.norm(values, axis=-1) + cfg.epsilon
    )
    diagnostics = {
        "mean_relative_delta": float(delta.mean()),
        "foreground_mean": float(fg.mean()),
        "persistence_mean": float(persist.mean()),
        "temporal_evidence_mean_norm": float(np.linalg.norm(temporal, axis=-1).mean()),
        "config": asdict(cfg),
    }
    return FeatureEnhancementOutput(
        features=enhanced.astype(values.dtype, copy=False),
        spatial_scale=spatial_scale.astype(np.float32, copy=False),
        temporal_evidence=temporal.astype(np.float32, copy=False),
        diagnostics=diagnostics,
    )
