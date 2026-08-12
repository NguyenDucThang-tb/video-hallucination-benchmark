from __future__ import annotations

import numpy as np

from src.methods.base import InferenceMethod, MethodOutput
from src.models.base import GenerationConfig


class DINOHealMethod(InferenceMethod):
    name = "dino_heal"

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> MethodOutput:
        if not hasattr(self.model, "generate_dino_heal"):
            raise RuntimeError(f"DINO-HEAL unsupported for adapter {self.model.name}")
        text, diagnostics = self.model.generate_dino_heal(
            video_frames, prompt, generation_config, self.config
        )
        if not diagnostics.get("dino_loaded", False):
            raise RuntimeError("DINOv2 failed to load; refusing to label CLIP-only fallback as DINO-HEAL")
        return MethodOutput(text, diagnostics)
