from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.benchmarks.base import BenchmarkLoader, BenchmarkSample


LETTERS = ("A", "B", "C", "D")
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def first_existing(record: dict, keys: list[str], default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def normalize_text(text) -> str:
    text = "" if text is None else str(text)
    return " ".join(text.strip().split()).lower()


def build_video_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        aliases = {
            path.name,
            path.stem,
            rel,
            Path(rel).with_suffix("").as_posix(),
        }
        for alias in aliases:
            index[str(alias)] = path
    return index


def find_video(video_id: str | None, index: dict[str, Path]) -> Path | None:
    if video_id is None:
        return None
    text = str(video_id).strip()
    path = Path(text)
    candidates = [text, path.name, path.stem, path.with_suffix("").name]
    if "_clip_" in text:
        candidates.append(text.split("_clip_", 1)[0])
    if "_" in text:
        left, right = text.split("_", 1)
        candidates.extend([f"{right}_{left}", right, left])
        if right.startswith("clip_"):
            candidates.append(left)
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    normalized = "".join(ch.lower() for ch in text if ch.isalnum())
    for candidate, candidate_path in index.items():
        if "".join(ch.lower() for ch in candidate if ch.isalnum()) == normalized:
            return candidate_path
    return None


def find_video_exact(video_id: str | None, index: dict[str, Path]) -> Path | None:
    """Resolve VidHalluc TSH/STH using the upstream exact filename semantics."""
    if video_id is None:
        return None
    text = str(video_id).strip()
    path = Path(text)
    for candidate in (text, path.name, path.stem):
        if candidate in index:
            return index[candidate]
    return None


def unresolved_video_path(root: Path, video_id: str | None) -> Path:
    name = Path(str(video_id or "unknown")).name
    if not Path(name).suffix:
        name = f"{name}.mp4"
    return root / "__unresolved__" / name


def build_ach_prompt(question: str, choices: dict[str, str]) -> str:
    lines = [
        question.strip(),
        "Choose exactly one answer from A, B, C, or D.",
        "Answer with only the letter.",
        "Choices:",
    ]
    for letter in LETTERS:
        if letter in choices:
            lines.append(f"{letter}. {choices[letter]}")
    return "\n".join(lines)


def build_bqa_prompt(question: str) -> str:
    return (
        f"{question.strip()}\n"
        "Answer using only 'yes' or 'no'."
    )


def build_tsh_prompt(question: str, output_protocol: str = "official") -> str:
    prompt = question + (
        "Sort these two actions in the order they occur in the video, and return which action "
        "happen before which one. If you only detect one action, return that action."
    )
    if output_protocol == "official":
        return prompt
    if output_protocol == "parser_compatible":
        return prompt + (
            " Respond with exactly one of: 'AB' if Action A happens before Action B; "
            "'BA' if Action B happens before Action A; 'A' if only Action A is visible; "
            "or 'B' if only Action B is visible. Do not include any other text."
        )
    raise ValueError(f"Unsupported VidHalluc TSH prompt protocol: {output_protocol!r}")


def build_sth_prompt() -> str:
    return (
        "Watch the given video and determine if a scene change occurs. "
        "If no change occurs, respond: 'Scene change: No, Locations: None'. "
        "If there is a scene change, respond in the format: "
        "'Scene change: Yes, Locations: from [location1] to [location2].'"
    )


class VidHallucLoader(BenchmarkLoader):
    def __init__(
        self,
        data_root: str | Path,
        tasks: list[str] | None = None,
        tsh_prompt_protocol: str = "official",
    ):
        self.video_root = Path(data_root)
        self.annotation_root = self._resolve_annotation_root(self.video_root)
        self.tasks = [task.lower() for task in (tasks or ["bqa", "mcq", "sth", "tsh"])]
        self.tsh_prompt_protocol = tsh_prompt_protocol
        self.video_index = build_video_index(self.video_root)

    def _resolve_annotation_root(self, video_root: Path) -> Path:
        if (video_root / "tsh.json").exists():
            return video_root
        if (video_root.parent / "tsh.json").exists():
            return video_root.parent
        return video_root

    def iter_samples(self) -> Iterable[BenchmarkSample]:
        for task in self.tasks:
            if task == "bqa":
                yield from self._iter_bqa()
            elif task in {"ach", "mcq"}:
                yield from self._iter_ach(task_name="mcq")
            elif task == "tsh":
                yield from self._iter_tsh()
            elif task == "sth":
                yield from self._iter_sth()
            else:
                raise ValueError(f"Unsupported VidHalluc task: {task}")

    def _iter_bqa(self) -> Iterable[BenchmarkSample]:
        data = json.loads((self.annotation_root / "ach_binaryqa.json").read_text(encoding="utf-8"))
        for section_id, section in data.items():
            if not isinstance(section, list):
                continue
            for question_index, item in enumerate(section):
                question = str(item.get("q", "")).strip()
                answers = item.get("a", {})
                if not question or not isinstance(answers, dict):
                    continue
                for clip_name, answer in answers.items():
                    video_path = find_video(clip_name, self.video_index)
                    gt = normalize_text(answer)
                    if gt not in {"yes", "no"}:
                        continue
                    video_resolved = video_path is not None
                    video_path = video_path or unresolved_video_path(self.video_root, clip_name)
                    yield BenchmarkSample(
                        sample_id=f"bqa:{section_id}:{question_index}:{clip_name}",
                        benchmark="vidhalluc",
                        task="bqa",
                        video_path=video_path,
                        prompt=build_bqa_prompt(question),
                        ground_truth=gt,
                        answer_type="yes_no",
                        metadata={
                            "source": "ach_binaryqa.json",
                            "section": section_id,
                            "question_index": question_index,
                            "video_name": clip_name,
                            "expected_clip_count": len(answers),
                            "video_resolved": video_resolved,
                        },
                    )

    def _iter_tsh(self) -> Iterable[BenchmarkSample]:
        data = json.loads((self.annotation_root / "tsh.json").read_text(encoding="utf-8"))
        for sample_id, item in data.items():
            video_path = find_video_exact(item.get("video"), self.video_index)
            gt = str(item.get("Correct Answer", "")).strip().upper()
            if gt not in {"AB", "BA"}:
                raise ValueError(f"Invalid TSH label for annotation {sample_id}: {gt!r}")
            video_resolved = video_path is not None
            video_path = video_path or unresolved_video_path(self.video_root, item.get("video"))
            yield BenchmarkSample(
                sample_id=f"tsh:{sample_id}",
                benchmark="vidhalluc",
                task="tsh",
                video_path=video_path,
                prompt=build_tsh_prompt(
                    str(item["Question"]),
                    output_protocol=self.tsh_prompt_protocol,
                ),
                ground_truth=gt,
                answer_type="ab_ba",
                metadata={
                    "source": "tsh.json",
                    "annotation_id": str(sample_id),
                    "video_name": item.get("video"),
                    "video_resolved": video_resolved,
                    "tsh_prompt_protocol": self.tsh_prompt_protocol,
                },
            )

    def _iter_sth(self) -> Iterable[BenchmarkSample]:
        data = json.loads((self.annotation_root / "sth.json").read_text(encoding="utf-8"))
        for video_id, item in data.items():
            video_path = find_video_exact(video_id, self.video_index)
            scene_change = str(item.get("Scene change", "")).strip().lower()
            locations = str(item.get("Locations", "")).strip()
            if scene_change not in {"yes", "no"}:
                raise ValueError(f"Invalid STH label for annotation {video_id}: {scene_change!r}")
            video_resolved = video_path is not None
            video_path = video_path or unresolved_video_path(self.video_root, video_id)
            yield BenchmarkSample(
                sample_id=f"sth:{video_id}",
                benchmark="vidhalluc",
                task="sth",
                video_path=video_path,
                prompt=build_sth_prompt(),
                ground_truth=scene_change,
                answer_type="text",
                metadata={
                    "source": "sth.json",
                    "annotation_id": str(video_id),
                    "video_name": video_id,
                    "scene_change": scene_change,
                    "locations": locations,
                    "video_resolved": video_resolved,
                },
            )

    def _iter_ach(self, task_name: str = "mcq") -> Iterable[BenchmarkSample]:
        data = json.loads((self.annotation_root / "ach_mcq.json").read_text(encoding="utf-8"))
        for section_id, section in data.items():
            for clip_name, item in section.items():
                video_path = find_video(clip_name, self.video_index)
                choices = {str(key).upper(): str(value) for key, value in item.get("Choices", {}).items()}
                gt = str(item.get("Correct Answer", "")).strip().upper()
                if gt not in LETTERS:
                    continue
                video_resolved = video_path is not None
                video_path = video_path or unresolved_video_path(self.video_root, clip_name)
                yield BenchmarkSample(
                    sample_id=f"{task_name}:{section_id}:{clip_name}",
                    benchmark="vidhalluc",
                    task=task_name,
                    video_path=video_path,
                    prompt=build_ach_prompt(str(item["Question"]), choices),
                    ground_truth=gt,
                    answer_type="mcq",
                    choices=choices,
                    metadata={
                        "source": "ach_mcq.json",
                        "section": section_id,
                        "video_name": clip_name,
                        "video_resolved": video_resolved,
                    },
                )
