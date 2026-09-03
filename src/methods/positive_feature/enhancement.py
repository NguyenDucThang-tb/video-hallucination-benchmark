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
    if str(device).startswith("cpu"):
        model = model.float()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    holder["_birefnet_model"] = model
    holder["_birefnet_transform"] = transform
    return model, transform


# def compute_birefnet_foreground(
#     video_frames,
#     T: int,
#     P: int,
#     birefnet_model,
#     birefnet_transform,
#     torch_module,
#     target_device,
# ) -> "torch.Tensor":
#     """Compute per-token foreground mask ``[T, P]`` using BiRefNet.

#     Args:
#         video_frames: Numpy array ``[n_frames, H, W, 3]``.
#         T: Number of temporal positions in the vision grid.
#         P: Number of spatial patches per frame (``Ht * Wt``).
#         birefnet_model: Loaded BiRefNet model.
#         birefnet_transform: Torchvision transform for BiRefNet input.
#         torch_module: The ``torch`` module reference.
#         target_device: Device to place the output tensor.

#     Returns:
#         Tensor of shape ``[T, P]`` with foreground scores in [0, 1].
#     """
#     from PIL import Image

#     n_frames = len(video_frames)
#     birefnet_parameter = next(birefnet_model.parameters())
#     birefnet_device = birefnet_parameter.device
#     birefnet_dtype = birefnet_parameter.dtype

#     images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in video_frames]
#     inputs = torch_module.stack([birefnet_transform(img) for img in images]).to(
#         device=birefnet_device,
#         dtype=birefnet_dtype,
#     )

#     with torch_module.inference_mode():
#         outputs = birefnet_model(inputs)
#         preds = outputs[-1].sigmoid() if isinstance(outputs, (list, tuple)) else outputs.sigmoid()

#     # Compute Ht, Wt from P (assume roughly square grid)
#     Ht = int(np.sqrt(P))
#     Wt = P // Ht
#     if Ht * Wt != P:
#         # Fallback: flatten to 1×P
#         Ht, Wt = 1, P

#     # Resize prediction masks to [n_frames, Ht, Wt]
#     preds = torch_module.nn.functional.interpolate(
#         preds, size=(Ht, Wt), mode="bilinear", align_corners=False,
#     ).squeeze(1)

#     scores = preds.view(n_frames, -1).float()  # [n_frames, P]

#     # Map n_frames → T temporal positions
#     if n_frames == T:
#         fg = scores
#     elif n_frames < T:
#         idx = np.rint(np.linspace(0, n_frames - 1, T)).astype(int)
#         fg = scores[idx]
#     else:
#         fg = torch_module.zeros(T, P, device=scores.device, dtype=scores.dtype)
#         for t in range(T):
#             lo = t * n_frames // T
#             hi = max(lo + 1, (t + 1) * n_frames // T)
#             fg[t] = scores[lo:hi].mean(0)

#     return fg.to(target_device)
def compute_birefnet_foreground(
    video_frames,
    T: int,
    P: int,
    birefnet_model,
    birefnet_transform,
    torch_module,
    target_device,
    thr: float = 0.15,
    kernel: int = 5,
    return_soft: bool = False,
    avg_weight: float = 0.7,
) -> "torch.Tensor":
    """
    Compute foreground evidence using BiRefNet and align it
    with visual temporal tokens.

    Parameters
    ----------
    video_frames:
        Numpy array [n_frames, H, W, 3] hoặc list[PIL.Image].

        Trường hợp quan trọng:
            n_frames == T
                -> mỗi temporal slice dùng 1 frame

            n_frames == 2*T
                -> temporal slice t dùng:
                    frame[2*t]
                    frame[2*t+1]

                foreground được hợp nhất bằng:
                    max(mask_0, mask_1)

    T: Number of temporal positions in vision grid.

    P: Number of spatial visual tokens per temporal slice:   P = Ht * Wt

    thr: Threshold dùng khi return_soft=False.

    kernel:   Morphological closing kernel.  <= 1 hoặc None để disable.

    return_soft:
        True:
            return float foreground confidence [0,1]

        False:
            return binary foreground mask.

    avg_weight:
        Hybrid pooling:

            g =
                avg_weight * avg_pool
                + (1 - avg_weight) * max_pool

    Returns
    -------
    fg:
        Tensor [T, P].

        return_soft=True:
            float32

        return_soft=False:
            bool
    """

    import numpy as np
    from PIL import Image

    F = torch_module.nn.functional

    # ============================================================
    # 0. Grid
    # ============================================================

    # Nếu caller có grid thật thì tốt nhất truyền Ht/Wt trực tiếp.
    # Với signature hiện tại chỉ có P nên suy ra gần-square grid.
    Ht = int(np.sqrt(P))
    Wt = P // Ht

    if Ht * Wt != P:
        # fallback an toàn
        Ht, Wt = 1, P

    n_frames = len(video_frames)

    if n_frames == 0:
        raise ValueError("video_frames is empty")

    # ============================================================
    # 1. Device / dtype của BiRefNet
    # ============================================================

    try:
        parameter = next(birefnet_model.parameters())
        birefnet_device = parameter.device
        birefnet_dtype = parameter.dtype
    except StopIteration:
        birefnet_device = target_device
        birefnet_dtype = torch_module.float32

    # ============================================================
    # 2. Convert frame -> PIL
    # ============================================================

    def to_pil(frame):

        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        arr = np.asarray(frame)

        # tránh lỗi nếu frame float [0,1]
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = arr * 255.0

        arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    # ============================================================
    # 3. BiRefNet prediction cho 1 frame
    # ============================================================

    def predict_one(frame):

        img = to_pil(frame)

        x = birefnet_transform(img).unsqueeze(0)

        x = x.to(
            device=birefnet_device,
            dtype=birefnet_dtype,
        )

        with torch_module.inference_mode():

            outputs = birefnet_model(x)

            if isinstance(outputs, (list, tuple)):
                pred = outputs[-1]
            else:
                pred = outputs

            pred = pred.sigmoid()

        # --------------------------------------------------------
        # Normalize shape về [H, W]
        # --------------------------------------------------------

        # thường là [1, 1, H, W]
        if pred.ndim == 4:
            pred = pred[0, 0]

        elif pred.ndim == 3:
            pred = pred[0]

        pred = (
            pred
            .float()
            .detach()
            .cpu()
        )

        return pred

    # ============================================================
    # 4. Predict tất cả frame trước
    #
    # Tránh chạy BiRefNet lại nếu một frame được sử dụng nhiều lần.
    # ============================================================

    all_preds = []

    for i in range(n_frames):
        all_preds.append(
            predict_one(video_frames[i])
        )

    # ============================================================
    # 5. Align frame -> visual temporal slice
    # ============================================================

    temporal_preds = []

    for t in range(T):

        # --------------------------------------------------------
        # CASE 1:
        #
        # 8 sampled frames -> T = 4
        #
        # temporal token t đại diện frame:
        #
        #       2t và 2t+1
        # --------------------------------------------------------

        if n_frames == 2 * T:
            idx0 = min(  2 * t,  n_frames - 1, )

            idx1 = min(  2 * t + 1, n_frames - 1, )

            pred0 = all_preds[idx0]
            pred1 = all_preds[idx1]

            # ====================================================
            # TEMPORAL RESCUE
            #
            # Nếu object bị BiRefNet miss ở một frame nhưng xuất
            # hiện rõ ở frame còn lại, vẫn giữ foreground evidence.
            # ====================================================

            pred = torch_module.maximum(  pred0,  pred1,   )

        # --------------------------------------------------------
        # CASE 2:
        # n_frames == T
        # --------------------------------------------------------

        elif n_frames == T:

            pred = all_preds[t]

        # --------------------------------------------------------
        # CASE 3:
        # fallback arbitrary number of frames
        # --------------------------------------------------------

        else:

            idx = round(
                t
                * (n_frames - 1)
                / max(T - 1, 1)
            )

            idx = min(
                max(idx, 0),
                n_frames - 1,
            )

            pred = all_preds[idx]

        # ========================================================
        # 6. Morphological closing
        # ========================================================

        if kernel is not None and kernel > 1:

            import cv2

            pred_np = pred.numpy().astype(
                np.float32,
                copy=False,
            )

            k = np.ones(
                (kernel, kernel),
                dtype=np.uint8,
            )

            pred_np = cv2.morphologyEx(
                pred_np,
                cv2.MORPH_CLOSE,
                k,
            )

            pred = torch_module.from_numpy(
                pred_np
            ).float()

        # ========================================================
        # 7. Pixel foreground -> visual token grid
        # ========================================================

        pred_4d = pred[
            None,
            None,
            ...,
        ]

        # --------------------------------------------------------
        # Average pooling
        #
        # phản ánh tỷ lệ foreground trong patch
        # --------------------------------------------------------

        g_avg = F.adaptive_avg_pool2d(
            pred_4d,
            output_size=(Ht, Wt),
        )[0, 0]

        # --------------------------------------------------------
        # Max pooling
        #
        # giúp không bỏ mất object nhỏ
        # --------------------------------------------------------

        g_max = F.adaptive_max_pool2d(
            pred_4d,
            output_size=(Ht, Wt),
        )[0, 0]

        # --------------------------------------------------------
        # Hybrid pooling
        # --------------------------------------------------------

        w_avg = float(avg_weight)
        w_max = 1.0 - w_avg

        g = (
            w_avg * g_avg
            + w_max * g_max
        )

        g = g.clamp(
            0.0,
            1.0,
        )

        # ========================================================
        # 8. Soft / Binary foreground
        # ========================================================

        if return_soft:

            foreground = (
                g
                .flatten()
                .float()
            )

        else:

            foreground = (
                g > thr
            ).flatten()

        temporal_preds.append(
            foreground
        )

    # ============================================================
    # 9. [T, P]
    # ============================================================

    fg = torch_module.stack(
        temporal_preds,
        dim=0,
    )

    if fg.shape != (T, P):
        raise RuntimeError(
            f"Foreground shape mismatch: "
            f"expected {(T, P)}, got {tuple(fg.shape)}"
        )

    if return_soft:
        return fg.to(
            device=target_device,
            dtype=torch_module.float32,
        )

    return fg.to(
        device=target_device,
        dtype=torch_module.bool,
    )

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


def enhance_tensor_by_frame_saliency(
    tensor,
    frame_saliency: np.ndarray,
    config,
    torch_module,
) -> tuple[Any, bool, dict]:
    """Compatibility wrapper for adapters that pass a feature tensor directly."""
    result, applied, diagnostics = enhance_output_by_frame_saliency(
        tensor,
        frame_saliency,
        config,
        torch_module,
    )
    if not applied:
        return result, applied, diagnostics

    n_frames = int(np.asarray(frame_saliency).size)
    token_count = int(tensor.numel() // tensor.shape[-1])
    tokens_per_frame = token_count // max(n_frames, 1)
    frame_major = (
        tensor.ndim == 3 and int(tensor.shape[0]) == n_frames
    ) or (
        tensor.ndim == 4
        and int(tensor.shape[0]) == 1
        and int(tensor.shape[1]) == n_frames
    )
    diagnostics = {
        **diagnostics,
        "positive_feature_frame_count": n_frames,
        "positive_feature_token_count": token_count,
        "positive_feature_tokens_per_frame": tokens_per_frame,
        "positive_feature_unaligned_token_count": token_count - tokens_per_frame * n_frames,
        "positive_feature_tensor_layout": "frame_major" if frame_major else "flattened_tokens",
    }
    return result, True, diagnostics


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
