from __future__ import annotations

COMPATIBILITY = {
    "llava-ov-7b": {
        "base": (True, "LLaVA-OV base adapter enabled for local real-video benchmark runs"),
        "tcd": (False, "Paper-based TCD reimplementation; post-audit real-video GPU validation is pending for LLaVA-OV"),
        "dino_heal": (True, "Local DINO fusion is not paper-faithful at patch level and lacks artifact-backed GPU validation"),
        "season": (False, "Paper-grounded local SEASON adapter implemented; real-video GPU smoke validation pending"),
    },
    "qwen2.5-vl-7b": {
        "base": (True, "Qwen2.5-VL base adapter enabled for local real-video benchmark runs"),
        "tcd": (True, "Paper-based TCD reimplementation validated with cached dual-branch decoding on Qwen2.5-VL/H200"),
        "dino_heal": (True, "Local DINO fusion enabled for Qwen2.5-VL local benchmark runs"),
        "season": (False, "Qwen SEASON path remains partial and is not the validated target adapter"),
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
