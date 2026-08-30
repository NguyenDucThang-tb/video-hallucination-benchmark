from __future__ import annotations

import numpy as np

from src.methods.base import InferenceMethod, MethodOutput
from src.models.base import GenerationConfig


class PositiveFeatureMethod(InferenceMethod):
    """Inference method that enhances visual features via a vision-encoder hook.

    Hooks into the vision encoder output during ``model.generate()`` to apply:
    1. Spatial saliency scaling using DINO foreground scores.
    2. Directed temporal motion evidence across consecutive frames.

    The method delegates all model-specific details (input preparation, grid
    computation, hook registration) to ``model.generate_positive_feature()``,
    keeping this class thin — exactly like ``DINOHealMethod``.
    """

    name = "positive_feature"

    def generate(
        self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig
    ) -> MethodOutput:
        if not hasattr(self.model, "generate_positive_feature"):
            raise RuntimeError(
                f"positive_feature unsupported: adapter {self.model.name} "
                "does not implement generate_positive_feature"
            )
        text, diagnostics = self.model.generate_positive_feature(
            video_frames, prompt, generation_config, self.config,
        )
        return MethodOutput(text, diagnostics)
