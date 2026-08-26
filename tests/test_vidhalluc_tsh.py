import json

import numpy as np

from src.benchmarks.vidhalluc.loader import VidHallucLoader, build_tsh_prompt
from src.methods.base import BaseMethod
from src.models.base import GenerationConfig, ModelAdapter
from scripts.run_benchmark import build_plan


OFFICIAL_SUFFIX = (
    "Sort these two actions in the order they occur in the video, and return which action "
    "happen before which one. If you only detect one action, return that action."
)


def test_tsh_prompt_preserves_action_order_and_official_instruction():
    question = "Action A. open the door\nAction B. sit down\n"
    prompt = build_tsh_prompt(question)
    assert prompt == question + OFFICIAL_SUFFIX
    assert prompt.index("Action A") < prompt.index("Action B")


def test_tsh_ab_ba_semantics_are_not_reversed(tmp_path):
    root = tmp_path / "vidhalluc"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "ab.mp4").write_bytes(b"")
    (data / "ba.mp4").write_bytes(b"")
    (root / "tsh.json").write_text(json.dumps({
        "one": {"video": "ab", "Question": "Action A. x\nAction B. y\n", "Correct Answer": "AB"},
        "two": {"video": "ba", "Question": "Action A. x\nAction B. y\n", "Correct Answer": "BA"},
    }))
    samples = list(VidHallucLoader(data, ["tsh"]).iter_samples())
    assert [sample.ground_truth for sample in samples] == ["AB", "BA"]


def test_tsh_loader_records_parser_compatible_prompt_protocol(tmp_path):
    root = tmp_path / "vidhalluc"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "ab.mp4").write_bytes(b"")
    (root / "tsh.json").write_text(json.dumps({
        "one": {"video": "ab", "Question": "Action A. x\nAction B. y\n", "Correct Answer": "AB"},
    }))

    sample = next(iter(VidHallucLoader(
        data,
        ["tsh"],
        tsh_prompt_protocol="parser_compatible",
    ).iter_samples()))

    assert "Respond with exactly one of" in sample.prompt
    assert sample.metadata["tsh_prompt_protocol"] == "parser_compatible"


class VisionProbeAdapter(ModelAdapter):
    name = "probe"
    checkpoint = "probe"

    def __init__(self):
        self.frames_seen = None
        self._diagnostics = []

    def generate(self, video_frames, prompt, generation_config):
        self.frames_seen = video_frames
        self._diagnostics = [{"vision_tensor_supplied": len(video_frames) > 0}]
        return "AB"

    def consume_generation_diagnostics(self, expected_count):
        return self._diagnostics


def test_base_method_passes_video_frames_to_model():
    adapter = VisionProbeAdapter()
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    output = BaseMethod(adapter).generate(frames, "question", GenerationConfig())
    assert adapter.frames_seen is frames
    assert output.diagnostics["vision_tensor_supplied"] is True


def test_base_and_tcd_jobs_keep_distinct_method_names():
    config = {
        "models": ["llava-ov-7b"],
        "methods": ["base", "tcd"],
        "benchmarks": [{"name": "vidhalluc", "tasks": ["tsh"]}],
    }
    plan = build_plan(config, allow_unvalidated=True)
    assert [job["method"] for job in plan] == ["base", "tcd"]
