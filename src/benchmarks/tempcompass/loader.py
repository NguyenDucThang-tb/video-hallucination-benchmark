from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from src.benchmarks.base import BenchmarkLoader, BenchmarkSample


TASK_FILES = {
    "multi-choice": "multi-choice.json",
    "yes_no": "yes_no.json",
    "caption_matching": "caption_matching.json",
    "captioning": "captioning.json",
}

ANSWER_SUFFIXES = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": "",
}

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _video_index(video_root: Path) -> dict[str, Path]:
    result = {}
    if not video_root.exists():
        return result
    for path in video_root.rglob("*"):
        if path.suffix.lower() in VIDEO_SUFFIXES:
            result.setdefault(path.stem, path)
    return result


def _unresolved_path(video_root: Path, video_id: str) -> Path:
    return video_root / "__unresolved__" / f"{video_id}.mp4"


def _mcq_choices(question: str) -> dict[str, str]:
    choices = {}
    for line in question.splitlines():
        match = re.match(r"^\s*([A-D])[.)]\s*(.+?)\s*$", line)
        if match:
            choices[match.group(1)] = match.group(2)
    return choices


def _answer_prefix(answer: str) -> str:
    match = re.match(r"^\s*([A-D])(?:[.)]|\s)", answer)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract TempCompass option from answer: {answer!r}")


def _caption_options(question: str) -> list[tuple[str, str, str]]:
    options = []
    for line in question.splitlines()[1:]:
        match = re.match(r"^\s*((?:Option|Sentence|Caption)\s+([A-D]|\d+)):\s*(.+?)\s*$", line)
        if match:
            options.append((match.group(1), match.group(2), match.group(3)))
    return options


def _caption_answer_key(answer: str, options: list[tuple[str, str, str]]) -> str:
    for label, short, sentence in options:
        if answer in {label, short, sentence, f"{label}: {sentence}"}:
            return label
    raise ValueError(f"Cannot match TempCompass caption answer to an option: {answer!r}")


class TempCompassLoader(BenchmarkLoader):
    """Load the official TempCompass question files without rewriting prompts."""

    def __init__(
        self,
        questions_root: str | Path,
        video_root: str | Path,
        meta_path: str | Path,
        tasks: list[str] | None = None,
    ):
        self.questions_root = Path(questions_root)
        self.video_root = Path(video_root)
        self.meta_path = Path(meta_path)
        self.tasks = tasks or list(TASK_FILES)
        unknown = set(self.tasks) - set(TASK_FILES)
        if unknown:
            raise ValueError(f"Unsupported TempCompass tasks: {sorted(unknown)}")
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.videos = _video_index(self.video_root)

    def iter_samples(self) -> Iterable[BenchmarkSample]:
        for task in self.tasks:
            yield from self._iter_task(task)

    def _iter_task(self, task: str) -> Iterable[BenchmarkSample]:
        path = self.questions_root / TASK_FILES[task]
        data = json.loads(path.read_text(encoding="utf-8"))
        for video_id, dimensions in data.items():
            video_path = self.videos.get(video_id)
            resolved = video_path is not None
            video_path = video_path or _unresolved_path(self.video_root, video_id)
            for aspect, questions in dimensions.items():
                dimension_meta = (self.meta.get(video_id, {}).get("eval_dim", {}).get(aspect) or {})
                fine_grained = "order" if aspect == "order" else dimension_meta.get("type")
                for question_index, item in enumerate(questions):
                    question = str(item["question"])
                    answer = str(item["answer"])
                    choices = {}
                    metadata = {
                        "source": TASK_FILES[task],
                        "video_id": video_id,
                        "video_resolved": resolved,
                        "temporal_aspect": aspect,
                        "fine_grained_aspect": fine_grained,
                        "question_index": question_index,
                        "official_question": question,
                        "official_answer": answer,
                        "official_answer_suffix": ANSWER_SUFFIXES[task],
                        "evaluation_protocol": "tempcompass_official_2024-03-23",
                    }

                    if task == "multi-choice":
                        choices = _mcq_choices(question)
                        ground_truth = _answer_prefix(answer)
                        answer_type = "tempcompass_multi_choice"
                    elif task == "yes_no":
                        ground_truth = answer.strip().lower()
                        answer_type = "tempcompass_yes_no"
                    elif task == "caption_matching":
                        caption_options = _caption_options(question)
                        ground_truth = _caption_answer_key(answer, caption_options)
                        answer_type = "tempcompass_caption_matching"
                        metadata["caption_options"] = [
                            {"label": label, "short": short, "sentence": sentence}
                            for label, short, sentence in caption_options
                        ]
                    else:
                        ground_truth = answer
                        answer_type = "tempcompass_captioning"
                        metadata["requires_llm_judge"] = True

                    yield BenchmarkSample(
                        sample_id=f"{task}:{video_id}:{aspect}:{question_index}",
                        benchmark="tempcompass",
                        task=task,
                        video_path=video_path,
                        prompt=question + ANSWER_SUFFIXES[task],
                        ground_truth=ground_truth,
                        answer_type=answer_type,
                        choices=choices,
                        metadata=metadata,
                    )
