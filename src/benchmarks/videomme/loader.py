from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

from src.benchmarks.base import BenchmarkLoader, BenchmarkSample


TASKS = {"short", "medium", "long"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PROMPT_PREFIX = (
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)


def _read_rows(root: Path) -> list[dict]:
    candidates = [
        root / "videomme.json", root / "questions.json", root / "data.json",
        root / "annotations.json", root / "test.json",
    ]
    candidates.extend(sorted(root.glob("*.jsonl")))
    candidates.extend(sorted(root.glob("*.csv")))
    candidates.extend(sorted(root.glob("*.json")))
    candidates.extend(sorted(root.rglob("*.parquet")))
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"No Video-MME annotation file found under {root}; expected videomme.json, "
            "questions.json, data.json, JSONL, CSV, or Parquet"
        )
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError(
                "Video-MME Parquet annotations require PyArrow. Install the project extra "
                "with `pip install -e '.[videomme]'`."
            ) from exc
        columns = parquet.read_table(path).to_pydict()
        names = list(columns)
        rows = [dict(zip(names, values)) for values in zip(*(columns[name] for name in names))]
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("data", value.get("questions", value.get("annotations", value)))
        if isinstance(value, dict):
            rows = list(value.values())
        elif isinstance(value, list):
            rows = value
        else:
            raise ValueError(f"Unsupported Video-MME annotation structure in {path}")
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"No valid Video-MME question rows in {path}")
    return rows


def _options(raw) -> tuple[dict[str, str], str]:
    if isinstance(raw, str):
        raw = [line.strip() for line in raw.splitlines() if line.strip()]
    if isinstance(raw, dict):
        choices = {str(key).upper(): str(value).strip() for key, value in raw.items()}
        text = "\n".join(f"{key}. {value}" for key, value in choices.items())
        return choices, text
    if not isinstance(raw, list):
        raise ValueError(f"Video-MME options must be a list or mapping, got {type(raw).__name__}")
    choices = {}
    lines = []
    for index, item in enumerate(raw):
        label = chr(ord("A") + index)
        content = str(item).strip()
        match = re.match(r"^([A-E])[.)]\s*(.*)$", content)
        if match:
            label, content = match.group(1), match.group(2).strip()
        choices[label] = content
        lines.append(f"{label}. {content}")
    return choices, "\n".join(lines)


def _video_index(root: Path) -> dict[str, Path]:
    index = {}
    video_root = root / "videos" if (root / "videos").is_dir() else root
    for path in video_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            index.setdefault(path.stem, path)
    return index


class VideoMMELoader(BenchmarkLoader):
    """Load Video-MME QA annotations with subtitle-free controlled prompting."""

    def __init__(self, data_root: str | Path | None, tasks: list[str] | None = None):
        if data_root is None:
            raise ValueError("Video-MME data_root must point to annotations and videos")
        self.root = Path(data_root)
        self.tasks = tasks or sorted(TASKS)
        unknown = set(self.tasks) - TASKS
        if unknown:
            raise ValueError(f"Unsupported Video-MME duration tasks: {sorted(unknown)}")
        self.rows = _read_rows(self.root)
        self.videos = _video_index(self.root)

    def iter_samples(self) -> Iterable[BenchmarkSample]:
        for row in self.rows:
            video_id = str(row["videoID"])
            duration = str(row["duration"]).strip().lower()
            if duration not in TASKS or duration not in self.tasks:
                continue
            question_id = str(row.get("question_id", row.get("questionID", "")))
            if not question_id:
                raise ValueError(f"Missing question_id for Video-MME row videoID={video_id}")
            choices, choice_text = _options(row["options"])
            answer = str(row["answer"]).strip().upper()
            if answer not in choices:
                raise ValueError(f"Answer {answer!r} is not in options for question {question_id}")
            question = str(row["question"]).strip()
            video_path = self.videos.get(video_id, self.root / "videos" / f"{video_id}.mp4")
            yield BenchmarkSample(
                sample_id=question_id,
                benchmark="videomme",
                task=duration,
                video_path=video_path,
                prompt=f"{PROMPT_PREFIX}\nQuestion: {question}\n{choice_text}\nThe best answer is:",
                ground_truth=answer,
                answer_type="mcq",
                choices=choices,
                metadata={
                    "video_id": video_id,
                    "video_resolved": video_id in self.videos,
                    "duration": duration,
                    "domain": row.get("domain"),
                    "sub_category": row.get("sub_category"),
                    "task_type": row.get("task_type"),
                    "official_question": question,
                    "official_options": row["options"],
                    "evaluation_protocol": "videomme_subtitle_free_controlled_8_uniform_frames",
                },
            )
