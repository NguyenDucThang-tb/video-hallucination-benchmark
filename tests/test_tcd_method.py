import numpy as np

from src.methods.tcd import TCDMethod
from src.models.base import GenerationConfig, StepOutput


class DummyTCDModel:
    name = "dummy"
    supports_step_logits = True

    def prepare_branch(self, video_frames, prompt: str, branch: str, **kwargs):
        return {"branch": branch}

    def decode_step(self, state, token_ids, output_attentions: bool = False):
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
