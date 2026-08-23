# Compatibility Matrix

"Algorithm" means the architecture can in principle expose the required
signals. "Runnable local" means the adapter can execute local-protocol Base
inference. It does not mean paper-compatible reproduction.

| Model | Method | Algorithm | Runnable local | Note |
|---|---|---:|---:|---|
| LLaVA-OV-7B | Base | yes | yes | local-protocol only; Base reproduction gate has not passed |
| LLaVA-OV-7B | TCD | yes | no | paper-based reimplementation; post-audit real-video GPU validation pending |
| LLaVA-OV-7B | DINO-HEAL | partial | no | frame-scalar token-norm saliency is not paper patch attention |
| LLaVA-OV-7B | SEASON | yes | no | paper-grounded adapter implemented; real-video GPU smoke validation pending |
| Qwen2.5-VL-7B | Base | yes | yes | local-protocol only; Base reproduction gate has not passed |
| Qwen2.5-VL-7B | TCD | yes | no | double-downsampling fixed; post-audit real-video GPU validation pending |
| Qwen2.5-VL-7B | DINO-HEAL | partial | no | frame-scalar token-norm saliency is not paper patch attention |
| Qwen2.5-VL-7B | SEASON | partial | no | input-only homogenization and unproven attention mapping; not the target adapter |
| LLaVA-Video-7B | Base | yes | no | checkpoint adapter and GPU validation pending |
| LLaVA-Video-7B | TCD | yes | no | needs synchronized token-level dual branch |
| LLaVA-Video-7B | DINO-HEAL | conditional | no | only if CLIP tower satisfies upstream contract |
| LLaVA-Video-7B | SEASON | yes | no | needs per-vision-layer homogenization hooks |

Numerical helper tests do not constitute model-method validation. All mitigation
methods are disabled until the Base gate passes and method-specific validation
is complete.
