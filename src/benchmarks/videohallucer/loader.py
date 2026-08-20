from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.benchmarks.base import BenchmarkLoader, BenchmarkSample


TASKS = {
    "orh": ("object_relation", "object_relation.json"),
    "tph": ("temporal", "temporal.json"),
    "sdh": ("semantic_detail", "semantic_detail.json"),
    "efh": ("external_factual", "external_factual.json"),
    "enfh": ("external_nonfactual", "external_nonfactual.json"),
}


class VideoHallucerLoader(BenchmarkLoader):
    def __init__(self, data_root: str | Path, tasks: list[str] | None = None):
        self.root = Path(data_root)
        self.tasks = tasks or list(TASKS)

    def iter_samples(self) -> Iterable[BenchmarkSample]:
        for task in self.tasks:
            folder, filename = TASKS[task]
            source = self.root / folder / filename
            rows = json.loads(source.read_text(encoding="utf-8"))
            for pair_index, row in enumerate(rows):
                pair_id = f"{task}:{pair_index}"
                for branch in ("basic", "hallucination"):
                    item = row[branch]
                    video_path = self.root / folder / "videos" / item["video"]
                    yield BenchmarkSample(
                        sample_id=f"{pair_id}:{branch}", benchmark="videohallucer", task=task,
                        video_path=video_path,
                        prompt=item["question"] + "\nAnswer the question using 'yes' or 'no'.",
                        ground_truth=item["answer"].lower(), answer_type="yes_no",
                        metadata={
                            "pair_id": pair_id,
                            "branch": branch,
                            "source": str(source),
                            "video_resolved": video_path.is_file(),
                        },
                    )
