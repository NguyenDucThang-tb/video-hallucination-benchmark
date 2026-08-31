import numpy as np

from src.models.qwen25_vl import Qwen25VLAdapter


class FakeAttentionTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_qwen_season_reduces_decoder_attention_to_native_video_frames():
    adapter = Qwen25VLAdapter.__new__(Qwen25VLAdapter)
    attention = FakeAttentionTensor(np.arange(16, dtype=np.float32).reshape(1, 1, 16, 1))

    scores = adapter._summarize_frame_attention((attention,), frame_count=8)

    assert scores.shape == (8,)
    np.testing.assert_allclose(scores.sum(), 1.0)
    assert np.all(np.isfinite(scores))

