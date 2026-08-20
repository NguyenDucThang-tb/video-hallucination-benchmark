import numpy as np

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
