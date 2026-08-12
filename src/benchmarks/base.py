from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    benchmark: str
    task: str
    video_path: Path
    prompt: str
    ground_truth: str
    answer_type: str
    choices: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkLoader:
    def iter_samples(self) -> Iterable[BenchmarkSample]:
        raise NotImplementedError
