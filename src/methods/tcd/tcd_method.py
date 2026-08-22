from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.methods.base import InferenceMethod, MethodOutput
from src.models.base import GenerationConfig


@dataclass(frozen=True)
class TCDConfig:
    alpha: float = 0.5
    beta: float = 0.5
    downsample_frames: int = 4
    threshold_space: str = "raw_logits"
    all_masked_behavior: str = "original_argmax"
    profile: bool = False

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("TCD alpha must be non-negative")
        if not 0 <= self.beta <= 1:
            raise ValueError("TCD beta must be within [0, 1]")
        if self.threshold_space != "raw_logits":
            raise ValueError("The EventHallusion paper defines the TCD threshold in raw-logit space")
        if self.all_masked_behavior not in {"original_argmax", "error"}:
            raise ValueError("all_masked_behavior must be 'original_argmax' or 'error'")


def chronological_downsample(frames: np.ndarray, count: int) -> tuple[np.ndarray, list[int]]:
    if not 1 <= count <= len(frames):
        raise ValueError("downsample count must be within the original frame count")
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int).tolist()
    return frames[indices], indices


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "device") and hasattr(value, "dtype")


def contrast_logits(original: Any, negative: Any, alpha: float, beta: float) -> Any:
    """Paper TCD: z=(1+a)z_ori-a*z_con, mask below b*max(z_ori)."""
    if _is_torch_tensor(original) or _is_torch_tensor(negative):
        if not (_is_torch_tensor(original) and _is_torch_tensor(negative)):
            raise TypeError("TCD branch logits must use the same tensor backend")
        if original.shape != negative.shape:
            raise ValueError("TCD branch logits must have the same shape")
        original = original.float()
        negative = negative.to(device=original.device, dtype=original.dtype)
        mixed = (1.0 + alpha) * original - alpha * negative
        threshold = beta * original.max()
        return mixed.masked_fill(mixed < threshold, float("-inf"))

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
        raw_config = dict(config or {})
        super().__init__(model, raw_config)
        tcd_config = {field: raw_config[field] for field in TCDConfig.__dataclass_fields__ if field in raw_config}
        self.tcd = TCDConfig(**tcd_config)

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> MethodOutput:
        if not self.model.supports_step_logits:
            raise RuntimeError(f"TCD unsupported: {self.model.name} adapter has no step logits")
        negative, negative_indices = chronological_downsample(video_frames, self.tcd.downsample_frames)
        original_state = self.model.prepare_branch(
            video_frames,
            prompt,
            branch="original",
            profile=self.tcd.profile,
            preserve_logits_on_device=True,
        )
        negative_state = self.model.prepare_branch(
            negative,
            prompt,
            branch="tcd_negative",
            profile=self.tcd.profile,
            preserve_logits_on_device=True,
        )
        generated: list[int] = []
        all_masked_fallbacks = 0
        original_decode_calls = 0
        negative_decode_calls = 0
        stopped_on_eos = False
        contrast_seconds = 0.0
        for _ in range(generation_config.max_new_tokens):
            ori = self.model.decode_step(original_state, generated)
            original_decode_calls += 1
            neg = self.model.decode_step(negative_state, generated)
            negative_decode_calls += 1
            contrast_started = time.perf_counter()
            logits = contrast_logits(ori.logits, neg.logits, self.tcd.alpha, self.tcd.beta)
            if _is_torch_tensor(logits):
                all_masked = bool(logits.isneginf().all().item())
            else:
                all_masked = bool(np.all(np.isneginf(logits)))
            if all_masked:
                if self.tcd.all_masked_behavior == "error":
                    raise RuntimeError("TCD masked every vocabulary token")
                logits = ori.logits
                all_masked_fallbacks += 1
            token = int(logits.argmax().item() if _is_torch_tensor(logits) else np.argmax(logits))
            contrast_seconds += time.perf_counter() - contrast_started
            generated.append(token)
            if getattr(self.model, "is_eos", lambda _: False)(token):
                stopped_on_eos = True
                break
        decode_started = time.perf_counter()
        text = self.model.decode_token_ids(generated)
        token_decode_seconds = time.perf_counter() - decode_started
        return MethodOutput(text, {
            "original_frame_count": int(len(video_frames)),
            "negative_frame_count": int(len(negative)),
            "negative_frame_positions": negative_indices,
            "original_decode_calls": original_decode_calls,
            "negative_decode_calls": negative_decode_calls,
            "all_masked_fallbacks": all_masked_fallbacks,
            "all_masked_behavior": self.tcd.all_masked_behavior,
            "generated_token_count": len(generated),
            "stopped_on_eos": stopped_on_eos,
            "contrast_seconds": contrast_seconds,
            "token_decode_seconds": token_decode_seconds,
            "original_branch": original_state.get("diagnostics", {}) if isinstance(original_state, dict) else {},
            "negative_branch": negative_state.get("diagnostics", {}) if isinstance(negative_state, dict) else {},
        })
