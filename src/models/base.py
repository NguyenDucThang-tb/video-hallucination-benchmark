from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0
    num_beams: int = 1

    def __post_init__(self) -> None:
        if self.do_sample or self.temperature != 0.0 or self.num_beams != 1:
            raise ValueError("Benchmark protocol requires greedy deterministic decoding")


@dataclass
class StepOutput:
    logits: Any
    frame_attention: Any | None = None
    cache: Any | None = None


class ModelAdapter(ABC):
    name: str
    checkpoint: str

    @abstractmethod
    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> str:
        raise NotImplementedError

    def prepare_branch(self, video_frames: np.ndarray, prompt: str, branch: str, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.name} does not expose branch preparation")

    def decode_step(self, state: Any, token_ids: list[int], output_attentions: bool = False) -> StepOutput:
        raise NotImplementedError(f"{self.name} does not expose token-level forward")

    def token_id_to_text(self, token_id: int) -> str:
        raise NotImplementedError

    @property
    def supports_step_logits(self) -> bool:
        return False

    @property
    def supports_frame_attention(self) -> bool:
        return False

    @property
    def supports_vision_layer_hooks(self) -> bool:
        return False
