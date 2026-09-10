from contextlib import nullcontext

import numpy as np
import pytest

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


def test_llava_video_positive_feature_uses_projector_patch_grid_and_reports_timing():
    torch = pytest.importorskip("torch")

    class FakeBiRefNet:
        def __init__(self):
            self.parameter = torch.zeros(1)
            self.batch_sizes = []

        def parameters(self):
            yield self.parameter

        def __call__(self, inputs):
            self.batch_sizes.append(len(inputs))
            return [inputs[:, :1]]

    class FakeModel:
        def __init__(self):
            self.mm_projector = torch.nn.Identity()
            self.projected = None

        def get_model(self):
            return self

        def generate(self, **kwargs):
            features = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) + 1
            self.projected = self.mm_projector(features)
            return torch.tensor([[7, 8]])

    class FakeTokenizer:
        def batch_decode(self, output_ids, skip_special_tokens):
            return ["answer"]

    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    frames[:, 0, 1] = 64
    frames[:, 1, 0] = 128
    frames[:, 1, 1] = 255
    birefnet = FakeBiRefNet()
    adapter = object.__new__(LlavaVideoAdapter)
    adapter.torch = torch
    adapter.device = torch.device("cpu")
    adapter.model = FakeModel()
    adapter.tokenizer = FakeTokenizer()
    adapter._ensure_birefnet_loaded = lambda checkpoint, device: (
        birefnet,
        lambda image: torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        .permute(2, 0, 1) / 255.0,
    )

    def build_inputs(video_frames, prompt):
        adapter._last_input_audit = {"video_modality_supplied": True}
        return {}

    adapter._build_inputs = build_inputs
    answer, diagnostics = adapter.generate_positive_feature(
        frames,
        "question",
        GenerationConfig(max_new_tokens=4),
        {"alpha": 0.2, "alpha_s": 0.0, "beta": 0.0, "birefnet_batch_size": 2},
    )

    assert answer == "answer"
    assert birefnet.batch_sizes == [2]
    assert diagnostics["positive_feature_mask_shape"] == [2, 4]
    assert diagnostics["foreground_spatial_std"] > 0
    assert diagnostics["positive_feature_hook_applied"] is True
    assert diagnostics["positive_feature_hook_call_count"] == 1
    assert diagnostics["positive_feature_output_sequence_token_count"] == 2
    assert diagnostics["positive_feature_saliency_seconds"] >= 0
    assert diagnostics["positive_feature_generate_seconds"] >= 0
    assert diagnostics["positive_feature_total_seconds"] >= 0


def test_llava_video_exposes_tcd_step_logits_contract():
    adapter = object.__new__(LlavaVideoAdapter)

    assert adapter.supports_step_logits is True


def test_llava_video_exposes_season_contract():
    adapter = object.__new__(LlavaVideoAdapter)

    assert adapter.supports_step_logits is True
    assert adapter.supports_frame_attention is True
    assert adapter.supports_vision_layer_hooks is True


def test_llava_video_branch_preparation_keeps_tcd_and_season_contracts():
    adapter = object.__new__(LlavaVideoAdapter)
    adapter._last_input_audit = {}
    adapter._build_inputs = lambda frames, prompt: {
        "inputs": np.array([[1, -200, 2]]),
        "attention_mask": np.ones((1, 3), dtype=np.int64),
        "images": [np.zeros((len(frames), 3, 4, 4))],
        "modalities": ["video"],
    }
    frames = np.zeros((8, 4, 4, 3), dtype=np.uint8)

    tcd = adapter.prepare_branch(frames, "question", "tcd_negative")
    season = adapter.prepare_branch(
        frames, "question", "original", attention_layers=(1, 2)
    )

    assert tcd["season_enabled"] is False
    assert season["season_enabled"] is True
    assert season["attention_layers"] == (1, 2)


def test_llava_video_multimodal_prefill_replaces_text_only_attention_mask():
    class _Model:
        def __init__(self):
            self.arguments = None

        def prepare_inputs_labels_for_multimodal(self, *args):
            self.arguments = args
            return (
                None,
                None,
                np.ones((1, 10), dtype=np.int64),
                None,
                np.zeros((1, 10, 4), dtype=np.float32),
                None,
            )

    adapter = object.__new__(LlavaVideoAdapter)
    adapter.model = _Model()
    state = {
        "model_inputs": {
            "input_ids": np.array([[1, -200, 2, 3]]),
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "images": [np.zeros((8, 3, 4, 4), dtype=np.float32)],
            "modalities": ["video"],
        },
        "diagnostics": {},
    }

    inputs_embeds, position_ids = adapter._prepare_multimodal_prefill(state)

    assert position_ids is None
    assert inputs_embeds.shape == (1, 10, 4)
    assert state["model_inputs"]["attention_mask"].shape == (1, 10)
    assert state["diagnostics"]["text_prompt_token_count"] == 4
    assert state["diagnostics"]["multimodal_prefill_token_count"] == 10
    assert adapter.model.arguments[6] == ["video"]
