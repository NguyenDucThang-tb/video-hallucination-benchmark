# Compatibility Matrix

"Algorithm" means the architecture can in principle expose the required
signals. "Runnable now" means this repository has a validated adapter. N/A is
reported unless both are true.

| Model | Method | Algorithm | Runnable now | Note |
|---|---|---:|---:|---|
| LLaVA-OV-7B | Base | yes | no | adapter exists; no committed real-video GPU validation artifact |
| LLaVA-OV-7B | TCD | yes | no | paper-based reimplementation; post-audit real-video GPU validation pending |
| LLaVA-OV-7B | DINO-HEAL | partial | no | frame-scalar token-norm saliency is not paper patch attention |
| LLaVA-OV-7B | SEASON | yes | no | paper-grounded adapter implemented; real-video GPU smoke validation pending |
| Qwen2.5-VL-7B | Base | yes | no | prior external runs are not committed validation artifacts |
| Qwen2.5-VL-7B | TCD | yes | no | double-downsampling fixed; post-audit real-video GPU validation pending |
| Qwen2.5-VL-7B | DINO-HEAL | partial | no | frame-scalar token-norm saliency is not paper patch attention |
| Qwen2.5-VL-7B | SEASON | partial | no | input-only homogenization and unproven attention mapping; not the target adapter |
| LLaVA-Video-7B | Base | yes | no | checkpoint adapter and GPU validation pending |
| LLaVA-Video-7B | TCD | yes | no | needs synchronized token-level dual branch |
| LLaVA-Video-7B | DINO-HEAL | conditional | no | only if CLIP tower satisfies upstream contract |
| LLaVA-Video-7B | SEASON | yes | no | needs per-vision-layer homogenization hooks |

Numerical helper tests do not constitute model-method validation. All pairs are
disabled until a real checkpoint and real video smoke artifact is committed.
