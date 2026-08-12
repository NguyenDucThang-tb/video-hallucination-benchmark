from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.methods.base import InferenceMethod, MethodOutput
from src.models.base import GenerationConfig


@dataclass(frozen=True)
class TCDConfig:
    alpha: float = 0.5
    beta: float = 0.5
    downsample_frames: int = 4
    threshold_space: str = "raw_logits"


def chronological_downsample(frames: np.ndarray, count: int) -> tuple[np.ndarray, list[int]]:
    if not 1 <= count <= len(frames):
        raise ValueError("downsample count must be within the original frame count")
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int).tolist()
    return frames[indices], indices


def contrast_logits(original: np.ndarray, negative: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Paper TCD: z=(1+a)z_ori-a*z_con, mask below b*max(z_ori)."""
    original = np.asarray(original, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if original.shape != negative.shape:
        raise ValueError("TCD branch logits must have the same shape")
    mixed = (1.0 + alpha) * original - alpha * negative
    threshold = beta * float(np.max(original))
    return np.where(mixed >= threshold, mixed, -np.inf)


class TCDMethod(InferenceMethod):
    name = "tcd"

    def __init__(self, model, config=None):
        super().__init__(model, config)
        self.tcd = TCDConfig(**(config or {}))

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> MethodOutput:
        if not self.model.supports_step_logits:
            raise RuntimeError(f"TCD unsupported: {self.model.name} adapter has no step logits")
        negative, negative_indices = chronological_downsample(video_frames, self.tcd.downsample_frames)
        original_state = self.model.prepare_branch(video_frames, prompt, branch="original")
        negative_state = self.model.prepare_branch(negative, prompt, branch="tcd_negative")
        generated: list[int] = []
        for _ in range(generation_config.max_new_tokens):
            ori = self.model.decode_step(original_state, generated)
            neg = self.model.decode_step(negative_state, generated)
            logits = contrast_logits(ori.logits, neg.logits, self.tcd.alpha, self.tcd.beta)
            if np.all(np.isneginf(logits)):
                logits = np.asarray(ori.logits)
            token = int(np.argmax(logits))
            generated.append(token)
            if getattr(self.model, "is_eos", lambda _: False)(token):
                break
        text = "".join(self.model.token_id_to_text(token) for token in generated)
        return MethodOutput(text, {"negative_frame_positions": negative_indices})
