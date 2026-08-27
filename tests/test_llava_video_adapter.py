from contextlib import nullcontext

import numpy as np

from src.models.base import GenerationConfig
from src.models.llava_video import LlavaVideoAdapter


class _Conversation:
    roles = ("user", "assistant")

    def __init__(self):
        self.messages = []

    def copy(self):
        return _Conversation()

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        return self.messages[0][1] + "\nASSISTANT:"


def test_llava_video_prompt_uses_single_video_token():
    adapter = object.__new__(LlavaVideoAdapter)
    adapter.conv_templates = {"qwen_1_5": _Conversation()}
    adapter.DEFAULT_IMAGE_TOKEN = "<image>"

    prompt = adapter._build_prompt("What happens first?")

    assert prompt.startswith("<image>\nWhat happens first?")
    assert prompt.count("<image>") == 1


def test_llava_video_path_resolver_prefers_existing_local_model_dir(tmp_path, monkeypatch):
    model_dir = tmp_path / "LLaVA-Video-7B-Qwen2"
    model_dir.mkdir()
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    adapter = object.__new__(LlavaVideoAdapter)

    assert adapter._resolve_model_path(None, "lmms-lab/LLaVA-Video-7B-Qwen2") == str(model_dir)


def test_llava_video_rejects_invalid_frame_shape_before_model_call():
    adapter = object.__new__(LlavaVideoAdapter)
    adapter._build_inputs = LlavaVideoAdapter._build_inputs.__get__(adapter)

    try:
        adapter._build_inputs(np.zeros((8, 32, 32), dtype=np.uint8), "question")
    except ValueError as exc:
        assert "[frames, height, width, 3]" in str(exc)
    else:
        raise AssertionError("invalid frame shape was accepted")


def test_llava_video_generate_uses_upstream_inputs_keyword():
    class _Model:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return np.array([[7, 8]])

    class _Tokenizer:
        def batch_decode(self, output_ids, skip_special_tokens):
            assert output_ids.tolist() == [[7, 8]]
            assert skip_special_tokens is True
            return ["answer"]

    adapter = object.__new__(LlavaVideoAdapter)
    adapter.model = _Model()
    adapter.tokenizer = _Tokenizer()
    adapter.torch = type("_Torch", (), {"inference_mode": staticmethod(nullcontext)})
    adapter._last_input_audit = {}
    adapter._build_inputs = lambda frames, prompt: {
        "inputs": np.array([[1, 2, 3]]),
        "attention_mask": np.array([[1, 1, 1]]),
        "images": [np.zeros((8, 3, 4, 4))],
        "modalities": ["video"],
    }

    answer = adapter.generate(
        np.zeros((8, 4, 4, 3), dtype=np.uint8),
        "question",
        GenerationConfig(),
    )

    assert answer == "answer"
    assert "inputs" in adapter.model.kwargs
    assert "input_ids" not in adapter.model.kwargs


def test_llava_video_exposes_tcd_step_logits_contract():
    adapter = object.__new__(LlavaVideoAdapter)

    assert adapter.supports_step_logits is True


def test_llava_video_exposes_season_contract():
    adapter = object.__new__(LlavaVideoAdapter)

    assert adapter.supports_step_logits is True
    assert adapter.supports_frame_attention is True
    assert adapter.supports_vision_layer_hooks is True
