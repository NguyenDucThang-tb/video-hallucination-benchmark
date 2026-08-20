from __future__ import annotations

COMPATIBILITY = {
    "llava-ov-7b": {
        "base": (False, "Adapter code exists, but this checkout has no artifact-backed real-video GPU validation"),
        "tcd": (False, "Local TCD reimplementation has no artifact-backed real-video GPU validation for LLaVA-OV"),
        "dino_heal": (False, "Local DINO fusion is not paper-faithful at patch level and lacks artifact-backed GPU validation"),
        "season": (False, "Local SEASON implementation is incomplete and not artifact-backed GPU validated"),
    },
    "qwen2.5-vl-7b": {
        "base": (False, "Adapter code exists, but prior Gadi runs are not committed as validation artifacts"),
        "tcd": (False, "Local TCD reimplementation lacks artifact-backed end-to-end validation and previously produced anomalous results"),
        "dino_heal": (False, "Local DINO fusion is not paper-faithful at patch level and lacks artifact-backed validation"),
        "season": (False, "Local SEASON implementation is incomplete and lacks artifact-backed validation"),
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
