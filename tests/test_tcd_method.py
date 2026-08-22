import numpy as np
import pytest

from src.methods.tcd import TCDMethod
from src.models.base import GenerationConfig, StepOutput


class DummyTCDModel:
    name = "dummy"
    supports_step_logits = True

    def prepare_branch(self, video_frames, prompt: str, branch: str, **kwargs):
        return {"branch": branch, "prefixes": []}

    def decode_step(self, state, token_ids, output_attentions: bool = False):
        state["prefixes"].append(tuple(token_ids))
        if state["branch"] == "original":
            return StepOutput(logits=np.array([0.1, 2.0, -1.0]))
        return StepOutput(logits=np.array([0.1, 0.2, -1.0]))

    def token_id_to_text(self, token_id: int) -> str:
        return ["A", "B", "<eos>"][token_id]

    def is_eos(self, token_id: int) -> bool:
        return token_id == 2

    def decode_token_ids(self, token_ids: list[int]) -> str:
        return "".join(self.token_id_to_text(token_id) for token_id in token_ids)


def test_tcd_method_prefers_contrasted_token():
    method = TCDMethod(DummyTCDModel(), {"alpha": 0.5, "beta": 0.0, "downsample_frames": 2})
    output = method.generate(np.zeros((4, 2, 2, 3), dtype=np.uint8), "q", GenerationConfig(max_new_tokens=1))
    assert output.text == "B"
    assert output.diagnostics["negative_frame_positions"] == [0, 3]


def test_tcd_branches_receive_the_same_generated_prefix():
    model = DummyTCDModel()
    states = []
    original_prepare = model.prepare_branch

    def capture(*args, **kwargs):
        state = original_prepare(*args, **kwargs)
        states.append(state)
        return state

    model.prepare_branch = capture
    TCDMethod(model, {"alpha": 0.5, "beta": 0.0, "downsample_frames": 2}).generate(
        np.zeros((4, 2, 2, 3), dtype=np.uint8), "q", GenerationConfig(max_new_tokens=2)
    )
    assert states[0]["prefixes"] == states[1]["prefixes"]


def test_tcd_all_masked_fallback_is_explicitly_counted():
    class AllMaskedModel(DummyTCDModel):
        def decode_step(self, state, token_ids, output_attentions: bool = False):
            return StepOutput(logits=np.array([-2.0, -3.0, -4.0]))

    output = TCDMethod(
        AllMaskedModel(),
        {"alpha": 0.5, "beta": 0.5, "downsample_frames": 2},
    ).generate(np.zeros((4, 2, 2, 3), dtype=np.uint8), "q", GenerationConfig(max_new_tokens=1))
    assert output.diagnostics["all_masked_fallbacks"] == 1
    assert output.diagnostics["all_masked_behavior"] == "original_argmax"


def test_tcd_all_masked_strict_mode_raises():
    class AllMaskedModel(DummyTCDModel):
        def decode_step(self, state, token_ids, output_attentions: bool = False):
            return StepOutput(logits=np.array([-2.0, -3.0, -4.0]))

    method = TCDMethod(
        AllMaskedModel(),
        {
            "alpha": 0.5,
            "beta": 0.5,
            "downsample_frames": 2,
            "all_masked_behavior": "error",
        },
    )
    with pytest.raises(RuntimeError, match="masked every"):
        method.generate(np.zeros((4, 2, 2, 3), dtype=np.uint8), "q", GenerationConfig(max_new_tokens=1))


class SpyModel:
    name = "spy"
    supports_step_logits = True

    def __init__(self):
        self.states = []
        self.decode_whole_sequence_calls = 0

    def prepare_branch(self, video_frames, prompt: str, branch: str, **kwargs):
        state = {
            "branch": branch,
            "frame_count": len(video_frames),
            "prefixes": [],
            "cache": None,
            "vision_calls": 0,
            "input_lengths": [],
            "attention_lengths": [],
            "diagnostics": {},
        }
        self.states.append(state)
        return state

    def decode_step(self, state, token_ids, output_attentions: bool = False):
        state["prefixes"].append(tuple(token_ids))
        if state["cache"] is None:
            state["vision_calls"] += 1
            state["cache"] = object()
            state["input_lengths"].append(10)
            state["attention_lengths"].append(10)
        else:
            state["input_lengths"].append(1)
            state["attention_lengths"].append(10 + len(token_ids))
        logits = np.array([0.0, 3.0, -2.0]) if not token_ids else np.array([0.0, 1.0, 4.0])
        return StepOutput(logits=logits, cache=state["cache"])

    def decode_token_ids(self, token_ids: list[int]) -> str:
        self.decode_whole_sequence_calls += 1
        return "decoded"

    def is_eos(self, token_id: int) -> bool:
        return token_id == 2


def test_tcd_cache_prefix_eos_and_vision_contract():
    model = SpyModel()
    output = TCDMethod(model, {"downsample_frames": 4, "beta": 0.0}).generate(
        np.zeros((8, 2, 2, 3), dtype=np.uint8),
        "same prompt",
        GenerationConfig(max_new_tokens=3),
    )
    original, negative = model.states
    assert [original["frame_count"], negative["frame_count"]] == [8, 4]
    assert original["prefixes"] == negative["prefixes"] == [(), (1,)]
    assert original["cache"] is not negative["cache"]
    assert original["vision_calls"] == negative["vision_calls"] == 1
    assert original["input_lengths"] == negative["input_lengths"] == [10, 1]
    assert original["attention_lengths"] == negative["attention_lengths"] == [10, 11]
    assert output.diagnostics["stopped_on_eos"] is True
    assert output.diagnostics["generated_token_count"] == 2
    assert model.decode_whole_sequence_calls == 1
    assert output.text == "decoded"


def test_tcd_generate_batch_runs_each_sample_once():
    model = SpyModel()
    outputs = TCDMethod(model, {"downsample_frames": 2, "beta": 0.0}).generate_batch(
        [np.zeros((4, 2, 2, 3), dtype=np.uint8)] * 2,
        ["first", "second"],
        GenerationConfig(max_new_tokens=1),
    )
    assert len(outputs) == 2
    assert len(model.states) == 4
