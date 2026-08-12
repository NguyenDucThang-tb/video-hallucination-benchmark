from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PredictionRecord:
    sample_id: str
    model: str
    method: str
    benchmark: str
    task: str
    prompt: str
    frame_indices: list[int]
    raw_output: str
    normalized_output: str | None
    ground_truth: str
    is_correct: bool | None
    parser_status: str
    error: str | None = None
    model_checkpoint: str | None = None
    upstream_commits: dict[str, str] = field(default_factory=dict)
    method_config: dict[str, Any] = field(default_factory=dict)
    sampling_config: dict[str, Any] = field(default_factory=dict)
    generation_config: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float | None = None
    peak_gpu_memory_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PredictionRecord":
        return cls(**value)

    @property
    def is_valid_for_resume(self) -> bool:
        return self.error is None and self.parser_status == "valid"
