from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.benchmarks.base import BenchmarkLoader, BenchmarkSample


SPLIT_VIDEO_DIRS = {
    "entire": ("entire",),
    "misleading": ("misleading",),
    "mix": ("mix", "interleave"),
    "interleave": ("interleave", "mix"),
}


def normalize_yes_no_label(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text


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
                    video_path = resolve_video_path(self.video_root, split, video_id)
                    yield BenchmarkSample(
                        sample_id=f"{split}:{video_id}:{index}", benchmark="eventhallusion", task=split,
                        video_path=video_path,
                        prompt=question["question"] + "\nPlease answer yes or no:",
                        ground_truth=normalize_yes_no_label(str(question["answer"])), answer_type="yes_no",
                        metadata={"event_info": video.get("event_info", {}), "source": str(source)},
                    )


def resolve_video_path(video_root: str | Path, split: str, video_id: str) -> Path:
    root = Path(video_root)
    try:
        candidates = SPLIT_VIDEO_DIRS[split]
    except KeyError as exc:
        raise ValueError(f"Unknown EventHallusion split: {split}") from exc
    for folder in candidates:
        candidate = root / folder / f"{video_id}.mp4"
        if candidate.is_file():
            return candidate
    return root / candidates[0] / f"{video_id}.mp4"


def description_reference(split: str, event_info: dict) -> str | None:
    if split in {"mix", "interleave"}:
        return event_info.get("unexpected")
    if split in {"entire", "misleading"}:
        return event_info.get("caption")
    raise ValueError(f"Unknown EventHallusion split: {split}")
