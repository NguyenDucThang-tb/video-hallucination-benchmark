from __future__ import annotations

from typing import Any

import numpy as np

from src.methods.season.positive_features import (
    FeatureEnhancementConfig,
    enhance_visual_features,
)


def enhance_tensor_by_frame_saliency(
    tensor: Any,
    frame_saliency: np.ndarray,
    config: FeatureEnhancementConfig,
    torch_module: Any,
) -> tuple[Any, bool, dict[str, Any]]:
    """Enhance visual tokens grouped in frame order.

    Projector outputs in the supported adapters are either ``[tokens, dim]``
    or ``[1, tokens, dim]``. Frame-level DINO saliency is broadcast over each
    frame's token group; temporal evidence is computed from the real tokens.
    """
    saliency = np.asarray(frame_saliency, dtype=np.float32).reshape(-1)
    n_frames = int(saliency.size)
    if not hasattr(tensor, "shape") or not hasattr(tensor, "detach") or n_frames < 2:
        return tensor, False, {}

    frame_view = None
    if tensor.ndim == 3 and int(tensor.shape[0]) == n_frames:
        frame_view = tensor
    elif tensor.ndim == 4 and int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == n_frames:
        frame_view = tensor[0]

    if frame_view is not None:
        tokens_per_frame = int(frame_view.shape[1])
        if tokens_per_frame <= 0:
            return tensor, False, {}
        features = frame_view.detach().float().cpu().numpy()
        foreground = np.broadcast_to(
            saliency[:, None],
            (n_frames, tokens_per_frame),
        )
        enhanced = enhance_visual_features(features, foreground, config)
        enhanced_tensor = torch_module.from_numpy(enhanced.features).to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
        result = tensor.clone()
        if result.ndim == 3:
            result[:] = enhanced_tensor
        else:
            result[0] = enhanced_tensor
        diagnostics = {
            **enhanced.diagnostics,
            "positive_feature_frame_count": n_frames,
            "positive_feature_token_count": n_frames * tokens_per_frame,
            "positive_feature_tokens_per_frame": tokens_per_frame,
            "positive_feature_unaligned_token_count": 0,
            "positive_feature_tensor_layout": "frame_major",
        }
        return result, True, diagnostics

    if tensor.ndim == 2:
        token_view = tensor
    elif tensor.ndim == 3 and int(tensor.shape[0]) == 1:
        token_view = tensor[0]
    else:
        return tensor, False, {}

    token_count = int(token_view.shape[0])
    if token_count < n_frames:
        return tensor, False, {}

    base, remainder = divmod(token_count, n_frames)
    spans: list[tuple[int, int]] = []
    start = 0
    for frame_index in range(n_frames):
        size = base + (1 if frame_index < remainder else 0)
        spans.append((start, start + size))
        start += size

    tokens_per_frame = min((end - begin for begin, end in spans), default=0)
    if tokens_per_frame <= 0:
        return tensor, False, {}

    features = np.stack(
        [
            token_view[begin : begin + tokens_per_frame]
            .detach()
            .float()
            .cpu()
            .numpy()
            for begin, _ in spans
        ],
        axis=0,
    )
    foreground = np.broadcast_to(
        saliency[:, None],
        (n_frames, tokens_per_frame),
    )
    enhanced = enhance_visual_features(features, foreground, config)
    enhanced_tensor = torch_module.from_numpy(enhanced.features).to(
        device=tensor.device,
        dtype=tensor.dtype,
    )

    result = tensor.clone()
    result_view = result if result.ndim == 2 else result[0]
    for frame_index, (begin, _) in enumerate(spans):
        result_view[begin : begin + tokens_per_frame] = enhanced_tensor[frame_index]

    diagnostics = {
        **enhanced.diagnostics,
        "positive_feature_frame_count": n_frames,
        "positive_feature_token_count": token_count,
        "positive_feature_tokens_per_frame": tokens_per_frame,
        "positive_feature_unaligned_token_count": token_count - tokens_per_frame * n_frames,
        "positive_feature_tensor_layout": "flattened_tokens",
    }
    return result, True, diagnostics


def enhance_output_by_frame_saliency(
    output: Any,
    frame_saliency: np.ndarray,
    config: FeatureEnhancementConfig,
    torch_module: Any,
) -> tuple[Any, bool, dict[str, Any]]:
    """Replace the first feature tensor found in a common model output type."""
    for attribute in ("last_hidden_state", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is None:
            continue
        enhanced, applied, diagnostics = enhance_tensor_by_frame_saliency(
            value, frame_saliency, config, torch_module
        )
        if applied:
            setattr(output, attribute, enhanced)
            return output, True, diagnostics

    if isinstance(output, (tuple, list)):
        updated = list(output)
        for index, item in enumerate(updated):
            enhanced, applied, diagnostics = enhance_output_by_frame_saliency(
                item, frame_saliency, config, torch_module
            )
            if applied:
                updated[index] = enhanced
                rebuilt = tuple(updated) if isinstance(output, tuple) else updated
                return rebuilt, True, diagnostics
        return output, False, {}

    return enhance_tensor_by_frame_saliency(
        output, frame_saliency, config, torch_module
    )
