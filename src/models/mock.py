from __future__ import annotations

import hashlib

import numpy as np

from .base import GenerationConfig, ModelAdapter


class DeterministicSmokeModel(ModelAdapter):
    """Pipeline smoke adapter. Its outputs are never research results."""

    name = "deterministic-smoke"
    checkpoint = "none"

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8") + video_frames.tobytes()).digest()
        return "yes" if digest[0] % 2 == 0 else "no"
