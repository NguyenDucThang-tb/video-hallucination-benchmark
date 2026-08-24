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
    video_reader: str = "opencv_robust_full_decode"

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


def vidhalluc_frame_indices(total_frames: int, fps: float, max_frames: int = 32) -> list[int]:
    """Reproduce VidHalluc's public one-frame-per-second then cap-at-32 policy."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    rounded_fps = round(fps)
    if rounded_fps <= 0:
        raise ValueError("VidHalluc sampling requires a positive rounded FPS")
    chosen = list(range(0, total_frames, rounded_fps))
    if len(chosen) > max_frames:
        chosen = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    return chosen


def sample_video(
    video_path: str | Path,
    num_frames: int = 8,
    strategy: str = "uniform",
    indices: list[int] | None = None,
) -> tuple[np.ndarray, FrameManifest]:
    if strategy not in {"uniform", "vidhalluc_official"}:
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
        reported_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()

    if reported_total <= 0:
        raise RuntimeError(f"Video reports no frames: {path}")

    requested = list(indices) if indices is not None else None
    decoded, effective_total, chosen = _decode_video_robust(
        path=path,
        num_frames=num_frames,
        strategy=strategy,
        fps=fps,
        requested_indices=requested,
        reported_total=reported_total,
        cv2_module=cv2,
    )

    manifest = FrameManifest(
        video_path=str(path),
        frame_indices=chosen,
        total_frames=effective_total,
        fps=fps,
        duration_seconds=(effective_total / fps if fps > 0 else 0.0),
        num_frames=len(chosen),
        strategy=strategy,
        short_video_policy=(
            "one_frame_per_second_no_padding"
            if strategy == "vidhalluc_official"
            else "repeat_nearest_linspace"
        ),
    )
    return np.stack(decoded), manifest


def _decode_video_robust(
    path: Path,
    num_frames: int,
    strategy: str,
    fps: float,
    requested_indices: list[int] | None,
    reported_total: int,
    cv2_module,
) -> tuple[list[np.ndarray], int, list[int]]:
    frames: list[np.ndarray] = []
    capture = cv2_module.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2RGB))
    finally:
        capture.release()

    actual_total = len(frames)
    if actual_total <= 0:
        raise RuntimeError(f"Could not decode any frame from {path}")

    effective_total = actual_total
    if requested_indices is None:
        chosen = (
            vidhalluc_frame_indices(actual_total, fps, num_frames)
            if strategy == "vidhalluc_official"
            else frame_indices(actual_total, num_frames)
        )
    else:
        if strategy == "uniform" and len(requested_indices) != num_frames:
            raise ValueError(f"Expected {num_frames} indices, got {len(requested_indices)}")
        if any(index < 0 for index in requested_indices):
            raise ValueError("Frame indices must be non-negative")
        if any(index >= actual_total for index in requested_indices):
            chosen = (
                vidhalluc_frame_indices(actual_total, fps, num_frames)
                if strategy == "vidhalluc_official"
                else frame_indices(actual_total, num_frames)
            )
        else:
            chosen = requested_indices

    if reported_total != actual_total:
        # Preserve the decodable frame count in the manifest; the raw file's
        # advertised count can be larger than what OpenCV can actually read.
        effective_total = actual_total

    return [frames[index] for index in chosen], effective_total, chosen
