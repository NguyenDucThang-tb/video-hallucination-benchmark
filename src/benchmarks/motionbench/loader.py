from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.benchmarks.base import BenchmarkLoader, BenchmarkSample


CATEGORY_SLUGS = {
    "Motion Recognition": "motion_recognition",
    "Location-related Motion": "location_related_motion",
    "Camera Motion": "camera_motion",
    "Motion-related Objects": "motion_related_objects",
    "Action Order": "action_order",
    "Repetition Count": "repetition_count",
}
SUPPORTED_TASKS = {"dev", "test", *CATEGORY_SLUGS.values()}
ANSWER_SUFFIX = "\nAnswer with only the option letter."
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _choices(question: str) -> dict[str, str]:
    return {
        match.group(1).upper(): match.group(2).strip()
        for match in re.finditer(r"(?m)^\s*([A-D])[.)]\s*(.+?)\s*$", question)
    }


def _video_index(video_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not video_root.exists():
        return index
    for path in video_root.rglob("*"):
        if path.suffix.lower() in VIDEO_SUFFIXES:
            index.setdefault(path.name, path)
    return index


class MotionBenchLoader(BenchmarkLoader):
    """Load MotionBench's official metadata and preserve its MCQ text."""

    def __init__(
        self,
        meta_path: str | Path,
        video_root: str | Path,
        tasks: list[str] | None = None,
    ):
        self.meta_path = Path(meta_path)
        self.video_root = Path(video_root)
        self.tasks = tasks or ["dev"]
        unknown = set(self.tasks) - SUPPORTED_TASKS
        if unknown:
            raise ValueError(f"Unsupported MotionBench tasks: {sorted(unknown)}")
        with self.meta_path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.videos = _video_index(self.video_root)
        self.expected = Counter()
        for row in self.rows:
            category = CATEGORY_SLUGS[row["question_type"]]
            for qa in row.get("qa", []):
                split = "test" if str(qa.get("answer", "")).strip().upper() == "NA" else "dev"
                self.expected[(split, category)] += 1
                self.expected[(split, split)] += 1

    def iter_samples(self) -> Iterable[BenchmarkSample]:
        for task in self.tasks:
            yield from self._iter_task(task)

    def _iter_task(self, task: str) -> Iterable[BenchmarkSample]:
        category_filter = task if task in CATEGORY_SLUGS.values() else None
        for row in self.rows:
            category_name = str(row["question_type"])
            category = CATEGORY_SLUGS[category_name]
            if category_filter is not None and category != category_filter:
                continue
            video_name = Path(str(row["video_path"])).name
            resolved_path = self.videos.get(video_name)
            video_path = resolved_path or self.video_root / "__unresolved__" / video_name
            for qa in row.get("qa", []):
                answer = str(qa.get("answer", "")).strip().upper()
                split = "test" if answer == "NA" else "dev"
                if category_filter is None and task != split:
                    continue
                if category_filter is not None and split != "dev":
                    continue
                output_task = category if category_filter is not None else task
                question = str(qa["question"])
                metadata = {
                    "source": str(self.meta_path),
                    "motionbench_uid": str(qa["uid"]),
                    "video_key": str(row.get("key", "")),
                    "video_name": video_name,
                    "video_resolved": resolved_path is not None,
                    "video_type": row.get("video_type"),
                    "question_type": category_name,
                    "question_category": category,
                    "evaluation_split": split,
                    "has_public_ground_truth": split == "dev",
                    "start": qa.get("start"),
                    "end": qa.get("end"),
                    "official_question": question,
                    "official_answer": answer,
                    "official_answer_suffix": ANSWER_SUFFIX,
                    "expected_task_records": self.expected[(split, output_task)],
                    "expected_split_records": self.expected[(split, split)],
                    "expected_category_records": self.expected[(split, category)],
                    "evaluation_protocol": "motionbench_official_mcq_controlled_16_frames",
                }
                if split == "test":
                    # Prevent private-label examples from being counted as wrong locally.
                    metadata["requires_llm_judge"] = True
                yield BenchmarkSample(
                    sample_id=str(qa["uid"]),
                    benchmark="motionbench",
                    task=output_task,
                    video_path=video_path,
                    prompt=question + ANSWER_SUFFIX,
                    ground_truth=answer,
                    answer_type="motionbench_mcq",
                    choices=_choices(question),
                    metadata=metadata,
                )
