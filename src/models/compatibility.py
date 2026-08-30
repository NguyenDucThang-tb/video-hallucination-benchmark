from __future__ import annotations

COMPATIBILITY = {
    "llava-ov-7b": {
        "base": (True, "LLaVA-OV base adapter enabled for local real-video benchmark runs"),
        "tcd": (False, "BASELINE NOT REPRODUCED; TCD grid and GPU validation are blocked"),
        "dino_heal": (False, "PARTIAL APPROXIMATION: frame-level fusion is not paper-faithful patch-level DINO-HEAL"),
        "season": (False, "NOT VALIDATED: Base gate and end-to-end GPU validation are pending"),
        "positive_feature": (False, "NOT VALIDATED: LLaVA-OV positive-feature projector hook requires an end-to-end GPU smoke test"),
    },
    "qwen2.5-vl-7b": {
        "base": (True, "Qwen2.5-VL base adapter enabled for local real-video benchmark runs"),
        "tcd": (False, "BASELINE NOT REPRODUCED; prior local H200 execution does not satisfy the reproduction gate"),
        "dino_heal": (False, "PARTIAL APPROXIMATION: frame-level scaling is not paper-faithful patch-level DINO-HEAL"),
        "season": (False, "PARTIAL: Qwen SEASON path has not passed end-to-end validation"),
        "positive_feature": (False, "NOT VALIDATED: Qwen positive-feature hook uses frame-level DINO saliency; run with --allow-unvalidated"),
    },
    "llava-video-7b": {
        "base": (False, "LLaVA Video checkpoint adapter not GPU-validated in this workspace"),
        "tcd": (False, "Token-level dual-branch adapter not GPU-validated"),
        "dino_heal": (False, "Conditional CLIP contract has not been GPU-validated"),
        "season": (False, "Vision homogenization and decoder attention adapter not GPU-validated"),
        "positive_feature": (False, "NOT VALIDATED: LLaVA-Video positive-feature projector hook requires an end-to-end GPU smoke test"),
    },
}


def check_compatibility(model: str, method: str) -> tuple[bool, str]:
    try:
        return COMPATIBILITY[model][method]
    except KeyError as exc:
        raise ValueError(f"Unknown model-method pair: {model}/{method}") from exc
