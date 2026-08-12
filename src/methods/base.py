from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.base import GenerationConfig, ModelAdapter


@dataclass
class MethodOutput:
    text: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


class InferenceMethod(ABC):
    name: str

    def __init__(self, model: ModelAdapter, config: dict[str, Any] | None = None):
        self.model = model
        self.config = config or {}

    @abstractmethod
    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> MethodOutput:
        raise NotImplementedError


class BaseMethod(InferenceMethod):
    name = "base"

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> MethodOutput:
        return MethodOutput(self.model.generate(video_frames, prompt, generation_config))
