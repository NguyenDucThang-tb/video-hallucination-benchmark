import numpy as np
import pytest

from src.methods.season.season_method import SeasonMethod
from src.models.base import GenerationConfig, StepOutput


class FakeSeasonAdapter:
    name = "fake-llava-ov"
    supports_step_logits = True
    supports_frame_attention = True
    supports_vision_layer_hooks = True

    def __init__(self):
        self.states = []
        self.enabled_layers = None

    def enable_season_attention(self, layers):
        self.enabled_layers = tuple(layers)

    def prepare_branch(self, video_frames, prompt, branch, **kwargs):
        state = {
            "branch": branch,
            "prefixes": [],
            "diagnostics": {"frame_count": len(video_frames)},
        }
        self.states.append(state)
        return state

    def decode_step(self, state, token_ids, output_attentions=False):
        state["prefixes"].append(tuple(token_ids))
        branch_logits = {
            "original": np.array([0.0, 3.0, 0.2]),
            "spatial_negative": np.array([0.0, 0.1, 2.0]),
            "temporal_homogenized": np.array([0.0, 0.2, 1.0]),
        }
        branch_attention = {
            "original": np.array([0.7, 0.1, 0.1, 0.1, 0, 0, 0, 0]),
            "spatial_negative": np.array([0.1, 0.7, 0.1, 0.1, 0, 0, 0, 0]),
            "temporal_homogenized": np.array([0.6, 0.2, 0.1, 0.1, 0, 0, 0, 0]),
        }
        return StepOutput(
            logits=branch_logits[state["branch"]],
            frame_attention=branch_attention[state["branch"]],
        )

    def decode_token_ids(self, token_ids):
        return "season-token" if token_ids else ""

    def is_eos(self, token_id):
        return False


def test_season_method_runs_three_synchronized_branches():
    model = FakeSeasonAdapter()
    output = SeasonMethod(model).generate(
        np.zeros((8, 2, 2, 3), dtype=np.uint8),
        "same prompt",
        GenerationConfig(max_new_tokens=2),
    )
    assert output.text == "season-token"
    assert [state["branch"] for state in model.states] == [
        "original", "spatial_negative", "temporal_homogenized"
    ]
    assert model.states[0]["prefixes"] == model.states[1]["prefixes"] == model.states[2]["prefixes"]
    assert output.diagnostics["generated_token_count"] == 2
    assert output.diagnostics["spatial_negative_mean_absolute_delta"] > 0
    assert len(output.diagnostics["token_diagnostics"]) == 2


def test_season_method_requires_exactly_eight_frames():
    with pytest.raises(ValueError, match="exactly 8"):
        SeasonMethod(FakeSeasonAdapter()).generate(
            np.zeros((7, 2, 2, 3), dtype=np.uint8),
            "prompt",
            GenerationConfig(max_new_tokens=1),
        )
