from pathlib import Path

import pytest

from src.benchmarks.eventhallusion.evaluator import description_reference
from src.benchmarks.eventhallusion.loader import resolve_video_path


def test_mix_uses_unexpected_event_not_caption():
    info = {"caption": "ordinary event", "unexpected": "unexpected event"}
    assert description_reference("mix", info) == "unexpected event"
    assert description_reference("interleave", info) == "unexpected event"
    assert description_reference("entire", info) == "ordinary event"


def test_unknown_split_fails():
    with pytest.raises(ValueError):
        description_reference("unknown", {})


def test_mix_falls_back_to_interleave_video_directory(tmp_path: Path):
    video = tmp_path / "interleave" / "123.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    assert resolve_video_path(tmp_path, "mix", "123") == video
