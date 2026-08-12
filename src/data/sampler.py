from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameManifest:
    video_path: str
    frame_indices: list[int]
    total_frames: int
    fps: float
    duration_seconds: float
    num_frames: int = 8
    strategy: str = "uniform"
    short_video_policy: str = "repeat_nearest_linspace"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frame_indices(total_frames: int, num_frames: int = 8) -> list[int]:
    """Return exactly num_frames deterministic nearest-linspace indices.

    Repetition is intentional when total_frames < num_frames. This keeps the
    model input shape fixed and makes the short-video policy auditable.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    return np.rint(np.linspace(0, total_frames - 1, num_frames)).astype(int).tolist()


def sample_video(
    video_path: str | Path,
    num_frames: int = 8,
    strategy: str = "uniform",
    indices: list[int] | None = None,
) -> tuple[np.ndarray, FrameManifest]:
    if strategy != "uniform":
        raise ValueError(f"Unsupported sampling strategy: {strategy}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for video sampling") from exc

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total <= 0:
            raise RuntimeError(f"Video reports no frames: {path}")
        chosen = list(indices) if indices is not None else frame_indices(total, num_frames)
        if len(chosen) != num_frames:
            raise ValueError(f"Expected {num_frames} indices, got {len(chosen)}")
        if any(i < 0 or i >= total for i in chosen):
            raise ValueError(f"Frame index outside [0, {total - 1}]")

        decoded: list[np.ndarray] = []
        for index in chosen:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Failed decoding frame {index} from {path}")
            decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    manifest = FrameManifest(
        video_path=str(path),
        frame_indices=chosen,
        total_frames=total,
        fps=fps,
        duration_seconds=(total / fps if fps > 0 else 0.0),
        num_frames=num_frames,
        strategy=strategy,
    )
    return np.stack(decoded), manifest
