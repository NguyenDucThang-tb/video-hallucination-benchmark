import numpy as np

from src.models.llava_ov import LlavaOVAdapter
from src.models.qwen25_vl import Qwen25VLAdapter


class FakeTensor:
    shape = (1, 2)

    def to(self, device):
        return self

    def sum(self, dim=None):
        return self

    def item(self):
        return 2


class RecordingProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return "<video>question"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": FakeTensor(),
            "attention_mask": FakeTensor(),
            "pixel_values_videos": FakeTensor(),
            "video_grid_thw": FakeTensor(),
        }


def _frames():
    return np.zeros((8, 4, 6, 3), dtype=np.uint8)


def test_llava_uses_one_video_placeholder():
    adapter = LlavaOVAdapter.__new__(LlavaOVAdapter)
    conversation, video = adapter._frames_to_conversation(_frames(), "question")

    assert conversation == [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "question"},
        ],
    }]
    assert video.shape == (8, 4, 6, 3)


def test_qwen_uses_one_video_placeholder():
    adapter = Qwen25VLAdapter.__new__(Qwen25VLAdapter)
    messages, video = adapter._frames_to_messages(_frames(), "question")

    assert messages == [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "question"},
        ],
    }]
    assert video.shape == (8, 4, 6, 3)


def test_video_helpers_reject_image_shaped_input():
    frames = np.zeros((4, 6, 3), dtype=np.uint8)
    for adapter, helper_name in (
        (LlavaOVAdapter.__new__(LlavaOVAdapter), "_frames_to_conversation"),
        (Qwen25VLAdapter.__new__(Qwen25VLAdapter), "_frames_to_messages"),
    ):
        try:
            getattr(adapter, helper_name)(frames, "question")
        except ValueError as error:
            assert "[frames, height, width, 3]" in str(error)
        else:
            raise AssertionError("image-shaped input should not be accepted as video")


def test_llava_processor_receives_video_not_images():
    adapter = LlavaOVAdapter.__new__(LlavaOVAdapter)
    adapter.processor = RecordingProcessor()
    adapter.device = "cpu"

    adapter._build_inputs(_frames(), "question")

    call = adapter.processor.calls[0]
    assert "images" not in call
    assert call["videos"][0].shape == (8, 4, 6, 3)
    assert adapter._last_input_audit["video_modality_supplied"] is True


def test_qwen_processor_receives_video_not_images():
    adapter = Qwen25VLAdapter.__new__(Qwen25VLAdapter)
    adapter.processor = RecordingProcessor()
    adapter.device = "cpu"

    adapter._prepare_inputs(_frames(), "question")

    call = adapter.processor.calls[0]
    assert "images" not in call
    assert call["videos"][0].shape == (8, 4, 6, 3)
    assert adapter._last_input_audit["video_grid_supplied"] is True
    assert adapter._last_input_audit["video_modality_supplied"] is True
