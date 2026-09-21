"""Shared experiment matrix for Qwen2.5-VL 32B/72B benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


MODELS = ("qwen2.5-vl-32b", "qwen2.5-vl-72b")
METHODS = ("base", "tcd", "dino_heal", "season", "positive_feature")

BENCHMARK_TASKS = {
    "vidhalluc": ("bqa", "mcq", "sth", "tsh"),
    "videohallucer": ("orh", "tph", "sdh", "efh", "enfh"),
    "eventhallusion": ("entire", "misleading", "mix"),
    "tempcompass": ("multi-choice", "yes_no", "caption_matching", "captioning"),
    "motionbench": (
        "motion_recognition",
        "location_related_motion",
        "camera_motion",
        "motion_related_objects",
        "action_order",
        "repetition_count",
    ),
}

EXPECTED_RECORDS = {
    "vidhalluc": {"bqa": 8550, "mcq": 4276, "sth": 445, "tsh": 600},
    "videohallucer": {
        "orh": 400,
        "tph": 352,
        "sdh": 400,
        "efh": 400,
        "enfh": 400,
    },
    "eventhallusion": {"entire": 114, "misleading": 102, "mix": 193},
    "tempcompass": {
        "multi-choice": 1580,
        "yes_no": 2453,
        "caption_matching": 1503,
        "captioning": 2004,
    },
    "motionbench": {
        "motion_recognition": 1478,
        "location_related_motion": 546,
        "camera_motion": 385,
        "motion_related_objects": 690,
        "action_order": 519,
        "repetition_count": 400,
    },
}


@dataclass(frozen=True)
class ResourceProfile:
    ngpus: int
    ncpus: int
    memory: str


RESOURCE_PROFILES = {
    "qwen2.5-vl-32b": ResourceProfile(ngpus=1, ncpus=12, memory="96GB"),
    "qwen2.5-vl-72b": ResourceProfile(ngpus=2, ncpus=24, memory="192GB"),
}


def safe_slug(value: str) -> str:
    return value.replace(".", "_").replace("-", "_")


def iter_matrix(
    models: tuple[str, ...] = MODELS,
    methods: tuple[str, ...] = METHODS,
    benchmarks: tuple[str, ...] | None = None,
    tasks: set[str] | None = None,
):
    selected_benchmarks = benchmarks or tuple(BENCHMARK_TASKS)
    for model in models:
        for method in methods:
            for benchmark in selected_benchmarks:
                for task in BENCHMARK_TASKS[benchmark]:
                    if tasks is None or task in tasks:
                        yield model, method, benchmark, task
