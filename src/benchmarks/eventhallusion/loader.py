from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.benchmarks.base import BenchmarkLoader, BenchmarkSample


class EventHallusionLoader(BenchmarkLoader):
    def __init__(self, questions_root: str | Path, video_root: str | Path, splits=None):
        self.questions_root = Path(questions_root)
        self.video_root = Path(video_root)
        self.splits = splits or ["entire", "misleading", "mix"]

    def iter_samples(self) -> Iterable[BenchmarkSample]:
        for split in self.splits:
            source = self.questions_root / f"{split}_questions.json"
            for video in json.loads(source.read_text(encoding="utf-8")):
                video_id = str(video["id"])
                for index, question in enumerate(video.get("questions", [])):
                    yield BenchmarkSample(
                        sample_id=f"{split}:{video_id}:{index}", benchmark="eventhallusion", task=split,
                        video_path=self.video_root / split / f"{video_id}.mp4",
                        prompt=question["question"] + "\nPlease answer yes or no:",
                        ground_truth=str(question["answer"]).lower(), answer_type="yes_no",
                        metadata={"event_info": video.get("event_info", {}), "source": str(source)},
                    )


def description_reference(split: str, event_info: dict) -> str | None:
    if split == "mix":
        return event_info.get("unexpected")
    if split in {"entire", "misleading"}:
        return event_info.get("caption")
    raise ValueError(f"Unknown EventHallusion split: {split}")
