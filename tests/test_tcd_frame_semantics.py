import numpy as np

from src.methods.tcd.tcd_method import TCDConfig, chronological_downsample


def test_negative_frame_count_is_a_target_count_not_a_stride():
    frames = np.arange(8)[:, None]
    sampled, positions = chronological_downsample(frames, 4)
    assert len(sampled) == 4
    assert positions == [0, 2, 5, 7]
    assert TCDConfig(negative_frame_count=4).negative_frame_count == 4
