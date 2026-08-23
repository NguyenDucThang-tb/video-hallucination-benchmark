from __future__ import annotations

from typing import Any


def unwrap_hidden_states(output: Any) -> Any:
    """Return a layer's hidden-state tensor without losing its output wrapper."""
    if isinstance(output, tuple):
        if not output:
            raise ValueError("Vision layer returned an empty tuple")
        return output[0]
    if isinstance(output, list):
        if not output:
            raise ValueError("Vision layer returned an empty list")
        return output[0]
    return output


def replace_hidden_states(output: Any, hidden_states: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    if isinstance(output, list):
        return [hidden_states, *output[1:]]
    return hidden_states


def frame_mean_context(hidden_states: Any, frame_count: int) -> Any:
    """Average equal-sized OneVision image groups while preserving tile/patch axes.

    LLaVA-OneVision processes each sampled frame as one or more equally-sized image
    crops. The leading dimension is therefore grouped as
    ``[frames, crops_per_frame, ...]``. This is an explicit adapter assumption and
    is validated before any averaging is applied.
    """
    if not hasattr(hidden_states, "shape") or hidden_states.ndim < 2:
        raise ValueError("Vision hidden states must expose at least two dimensions")
    leading = int(hidden_states.shape[0])
    if leading % frame_count != 0:
        raise ValueError(
            f"Vision batch dimension {leading} is not divisible by frame_count={frame_count}"
        )
    crops_per_frame = leading // frame_count
    grouped = hidden_states.reshape(frame_count, crops_per_frame, *hidden_states.shape[1:])
    if grouped.__class__.__module__.startswith("torch"):
        return grouped.mean(dim=0, keepdim=True).expand_as(grouped).reshape_as(hidden_states)

    import numpy as np

    context = grouped.mean(axis=0, keepdims=True)
    return np.broadcast_to(context, grouped.shape).reshape(hidden_states.shape)


def blend_temporal_hidden(hidden_states: Any, original_context: Any, beta: float) -> Any:
    if hidden_states.shape != original_context.shape:
        raise ValueError(
            "Temporal branch and original layer context must share shape: "
            f"{tuple(hidden_states.shape)} != {tuple(original_context.shape)}"
        )
    if hidden_states.__class__.__module__.startswith("torch"):
        original_context = original_context.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    return (1.0 - beta) * hidden_states + beta * original_context
