import numpy as np

from src.data.sampler import frame_indices, sample_video, vidhalluc_frame_indices


def test_uniform_indices_are_deterministic():
    assert frame_indices(100, 8) == [0, 14, 28, 42, 57, 71, 85, 99]
    assert frame_indices(100, 8) == frame_indices(100, 8)


def test_short_video_repeats_indices():
    assert frame_indices(3, 8) == [0, 0, 1, 1, 1, 1, 2, 2]


def test_vidhalluc_sampling_uses_one_frame_per_second_and_caps_at_32():
    assert vidhalluc_frame_indices(100, 25.0, 32) == [0, 25, 50, 75]
    indices = vidhalluc_frame_indices(1000, 25.0, 32)
    assert len(indices) == 32
    assert indices[0] == 0
    assert indices[-1] == 999


def test_sample_short_video_returns_exactly_eight(tmp_path):
    import cv2

    path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 16))
    assert writer.isOpened()
    for value in (0, 80, 160):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()
    frames, manifest = sample_video(path)
    assert frames.shape == (8, 16, 16, 3)
    assert manifest.frame_indices == [0, 0, 1, 1, 1, 1, 2, 2]
    assert manifest.total_frames == 3


def test_sample_video_supports_non_monotonic_repeated_indices(tmp_path):
    import cv2

    path = tmp_path / "manual.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (8, 8))
    assert writer.isOpened()
    for value in (0, 40, 80, 120):
        writer.write(np.full((8, 8, 3), value, dtype=np.uint8))
    writer.release()

    frames, manifest = sample_video(path, num_frames=4, indices=[3, 0, 3, 1])
    assert frames.shape == (4, 8, 8, 3)
    assert manifest.frame_indices == [3, 0, 3, 1]
    assert int(frames[0, 0, 0, 0]) > int(frames[1, 0, 0, 0])
    assert int(frames[2, 0, 0, 0]) == int(frames[0, 0, 0, 0])
