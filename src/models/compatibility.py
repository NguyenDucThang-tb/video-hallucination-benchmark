from __future__ import annotations

COMPATIBILITY = {
    "llava-ov-7b": {
        "base": (True, "LLaVA-OneVision adapter enabled locally; validate with a smoke run before full benchmark"),
        "tcd": (True, "Token-level dual-branch adapter enabled for LLaVA-OV; validate with a smoke run before full benchmark"),
        "dino_heal": (False, "Upstream patch targets CLIP Video-LLaVA; OV architecture differs"),
        "season": (False, "Vision homogenization and decoder attention adapter not GPU-validated"),
    },
    "qwen2.5-vl-7b": {
        "base": (True, "Offline checkpoint load validated on Gadi; base generation adapter enabled"),
        "tcd": (True, "Token-level dual-branch adapter enabled for Qwen; validate with a smoke run before full benchmark"),
        "dino_heal": (False, "No validated DINO saliency fusion for Qwen visual encoder"),
        "season": (False, "Qwen vision and decoder attention adapter not GPU-validated"),
    },
    "llava-video-7b": {
        "base": (False, "LLaVA Video checkpoint adapter not GPU-validated in this workspace"),
        "tcd": (False, "Token-level dual-branch adapter not GPU-validated"),
        "dino_heal": (False, "Conditional CLIP contract has not been GPU-validated"),
        "season": (False, "Vision homogenization and decoder attention adapter not GPU-validated"),
    },
}


def check_compatibility(model: str, method: str) -> tuple[bool, str]:
    try:
        return COMPATIBILITY[model][method]
    except KeyError as exc:
        raise ValueError(f"Unknown model-method pair: {model}/{method}") from exc
