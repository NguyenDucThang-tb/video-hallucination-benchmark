"""Positive visual-feature enhancement using BiRefNet foreground segmentation.

This module implements the core enhancement pipeline that can be called
from any model adapter's vision encoder hook:

Spatial Enhancement:
    F[t,p]  = foreground mask from BiRefNet (per token per frame)
    P[p]    = persistence map = (1/T) * Σ_t F[t,p]
    S       = α·F + α_s·P
    V_spatial = V ⊙ (1 + S)

Temporal Enhancement:
    Diff[t,p]  = V[t,p] - V[t-1,p]  (frame-to-frame change)
    Diff       = Diff * F            (mask out background)
    Diff       = (Diff / ||Diff||) * ||V||  (norm-stabilize)

Fusion:
    V' = V·(1+S) + β·Diff
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PositiveFeatureConfig:
    """Configuration for BiRefNet-based positive feature enhancement."""

    alpha: float = 0.4
    """Weight for foreground saliency scaling."""

    alpha_s: float = 0.4
    """Weight for foreground persistence scaling."""

    beta: float = 0.4
    """Weight for directed temporal motion evidence."""

    epsilon: float = 1e-6
    """Small constant to avoid division by zero."""

    use_birefnet: bool = True
    """If True, use BiRefNet for foreground; if False, fall back to DINO."""

    birefnet_checkpoint: str = "ZhengPeng7/BiRefNet"
    """HuggingFace checkpoint for BiRefNet model."""

    dino_checkpoint: str = "facebook/dinov2-large"
    """HuggingFace checkpoint for DINO model (fallback)."""

    saliency_device: str = "cpu"
    """Device to run the saliency model on."""


def ensure_birefnet_loaded(holder: dict, checkpoint: str, device: str, torch_module):
    """Lazy-load BiRefNet model and transform, caching on *holder*.

    Args:
        holder: A dict (typically adapter ``self.__dict__``) to cache
            ``_birefnet_model`` and ``_birefnet_transform``.
        checkpoint: HuggingFace checkpoint string.
        device: Device string (``"cpu"`` or ``"cuda"``).
        torch_module: The ``torch`` module reference.

    Returns:
        (birefnet_model, birefnet_transform) tuple.
    """
    if holder.get("_birefnet_model") is not None:
        return holder["_birefnet_model"], holder["_birefnet_transform"]

    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms

    model = AutoModelForImageSegmentation.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    ).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    holder["_birefnet_model"] = model
    holder["_birefnet_transform"] = transform
    return model, transform


def compute_birefnet_foreground(
    video_frames,
    T: int,
    P: int,
    birefnet_model,
    birefnet_transform,
    torch_module,
    target_device,
) -> "torch.Tensor":
    """Compute per-token foreground mask ``[T, P]`` using BiRefNet.

    Args:
        video_frames: Numpy array ``[n_frames, H, W, 3]``.
        T: Number of temporal positions in the vision grid.
        P: Number of spatial patches per frame (``Ht * Wt``).
        birefnet_model: Loaded BiRefNet model.
        birefnet_transform: Torchvision transform for BiRefNet input.
        torch_module: The ``torch`` module reference.
        target_device: Device to place the output tensor.

    Returns:
        Tensor of shape ``[T, P]`` with foreground scores in [0, 1].
    """
    from PIL import Image

    n_frames = len(video_frames)
    birefnet_device = next(birefnet_model.parameters()).device

    images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in video_frames]
    inputs = torch_module.stack([birefnet_transform(img) for img in images]).to(birefnet_device)

    with torch_module.inference_mode():
        outputs = birefnet_model(inputs)
        preds = outputs[-1].sigmoid() if isinstance(outputs, (list, tuple)) else outputs.sigmoid()

    # Compute Ht, Wt from P (assume roughly square grid)
    Ht = int(np.sqrt(P))
    Wt = P // Ht
    if Ht * Wt != P:
        # Fallback: flatten to 1×P
        Ht, Wt = 1, P

    # Resize prediction masks to [n_frames, Ht, Wt]
    preds = torch_module.nn.functional.interpolate(
        preds, size=(Ht, Wt), mode="bilinear", align_corners=False,
    ).squeeze(1)

    scores = preds.view(n_frames, -1).float()  # [n_frames, P]

    # Map n_frames → T temporal positions
    if n_frames == T:
        fg = scores
    elif n_frames < T:
        idx = np.rint(np.linspace(0, n_frames - 1, T)).astype(int)
        fg = scores[idx]
    else:
        fg = torch_module.zeros(T, P, device=scores.device, dtype=scores.dtype)
        for t in range(T):
            lo = t * n_frames // T
            hi = max(lo + 1, (t + 1) * n_frames // T)
            fg[t] = scores[lo:hi].mean(0)

    return fg.to(target_device)


def enhance_visual_embeddings(
    V: "torch.Tensor",
    fg: "torch.Tensor",
    config: PositiveFeatureConfig,
    torch_module,
) -> tuple["torch.Tensor", dict[str, Any]]:
    """Apply spatial + temporal enhancement to visual token embeddings.

    This is the core computation shared by all model adapters.

    Args:
        V: Visual embeddings shaped ``[T, P, D]``.
        fg: Foreground mask shaped ``[T, P]`` with values in [0, 1].
        config: Enhancement hyperparameters.
        torch_module: The ``torch`` module reference.

    Returns:
        (enhanced_V, diagnostics) where enhanced_V has the same shape as V.

    Formula:
        persist = mean_t(fg)  →  [P]  broadcast to [T, P]
        S = α·fg + α_s·persist
        V_spatial = V ⊙ (1 + S)
        Diff[t] = V[t] - V[t-1]  (zero for t=0)
        Diff = Diff * fg          (mask background)
        Diff = (Diff / ||Diff||) * ||V||   (norm-stabilize)
        V' = V_spatial + β·Diff
    """
    T, P, D = V.shape
    eps = config.epsilon

    # -- Spatial --
    persist = fg.mean(0, keepdim=True).expand(T, P)  # [T, P]
    S = config.alpha * fg + config.alpha_s * persist  # [T, P]
    V_spatial = V * (1.0 + S.unsqueeze(-1))           # [T, P, D]

    # -- Temporal --
    diff = torch_module.zeros_like(V)
    diff[1:] = V[1:] - V[:-1]
    diff = diff * fg.unsqueeze(-1)  # mask background

    # Norm-stabilize: Diff = (Diff / ||Diff||) * ||V||
    diff_norm = diff.norm(dim=-1, keepdim=True)  # [T, P, 1]
    v_norm = V.norm(dim=-1, keepdim=True)        # [T, P, 1]
    diff = diff / (diff_norm + eps) * v_norm

    # -- Fusion --
    V_prime = V_spatial + config.beta * diff

    # Diagnostics
    delta = ((V_prime - V).norm(dim=-1) / (V.norm(dim=-1) + eps)).mean().item()
    diagnostics = {
        "positive_feature_delta": float(delta),
        "foreground_mean": float(fg.mean().item()),
        "persistence_mean": float(persist.mean().item()),
        "temporal_evidence_mean_norm": float(diff.norm(dim=-1).mean().item()),
        "alpha": config.alpha,
        "alpha_s": config.alpha_s,
        "beta": config.beta,
    }
    return V_prime, diagnostics


def enhance_output_by_frame_saliency(
    output,
    frame_saliency: np.ndarray,
    config,
    torch_module,
) -> tuple[Any, bool, dict]:
    """Wrapper for hook usage with frame-level (not patch-level) saliency.

    This is used by adapters that only have frame-level DINO saliency
    (not per-patch BiRefNet masks). It broadcasts frame scores across
    all patches, then applies the spatial+temporal enhancement.

    Args:
        output: The vision encoder / projector output (tensor or structured).
        frame_saliency: 1-D array of per-frame saliency scores ``[n_frames]``.
        config: ``FeatureEnhancementConfig`` or ``PositiveFeatureConfig``.
        torch_module: The ``torch`` module reference.

    Returns:
        (modified_output, applied_bool, diagnostics_dict)
    """
    # Extract tensor from output
    tensor = _coerce_feature_tensor(output)
    if tensor is None:
        return output, False, {}

    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    f = tensor.reshape(-1, tensor.shape[-1]).float()
    n_vis, D = f.shape

    n_frames = len(frame_saliency)
    if n_frames == 0 or n_vis == 0:
        return output, False, {}

    P = max(n_vis // n_frames, 1)
    T = n_frames

    # If total tokens don't divide evenly, skip
    if T * P != n_vis:
        # Try to find a reasonable P
        P = n_vis // T
        if T * P != n_vis:
            return output, False, {"skip_reason": f"n_vis={n_vis} not divisible by T={T}"}

    V = f.view(T, P, D)

    # Broadcast frame saliency to [T, P]
    fg_np = np.asarray(frame_saliency, dtype=np.float32)
    fg = torch_module.as_tensor(fg_np, device=V.device, dtype=V.dtype)
    fg = fg.unsqueeze(-1).expand(T, P)  # [T, P]

    # Build config
    alpha = getattr(config, "alpha", 0.4)
    alpha_s = getattr(config, "alpha_spatial", getattr(config, "alpha_s", 0.4))
    beta = getattr(config, "beta_temporal", getattr(config, "beta", 0.4))
    epsilon = getattr(config, "epsilon", 1e-6)

    pf_config = PositiveFeatureConfig(
        alpha=float(alpha),
        alpha_s=float(alpha_s),
        beta=float(beta),
        epsilon=float(epsilon),
    )

    V_prime, diagnostics = enhance_visual_embeddings(V, fg, pf_config, torch_module)

    result = V_prime.reshape(orig_shape).to(orig_dtype)

    # Put result back into the output structure
    modified_output = _replace_feature_tensor(output, result)
    return modified_output, True, diagnostics


def _coerce_feature_tensor(value):
    """Extract the main feature tensor from a vision encoder output."""
    if value is None:
        return None
    if hasattr(value, "last_hidden_state") and value.last_hidden_state is not None:
        return value.last_hidden_state
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    if isinstance(value, (tuple, list)):
        for item in value:
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                return item
        return None
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return value
    return None


def _replace_feature_tensor(output, new_tensor):
    """Replace the main feature tensor inside a vision encoder output structure."""
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        output.last_hidden_state = new_tensor
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        output.pooler_output = new_tensor
        return output
    if isinstance(output, tuple):
        output_list = list(output)
        for idx, item in enumerate(output_list):
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                output_list[idx] = new_tensor
                return tuple(output_list)
        return output
    return new_tensor
