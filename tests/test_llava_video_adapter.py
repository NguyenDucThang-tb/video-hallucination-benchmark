import numpy as np

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
