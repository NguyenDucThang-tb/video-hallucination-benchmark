# Compatibility Matrix

"Algorithm" means the architecture can in principle expose the required
signals. "Runnable now" means this repository has a validated adapter. N/A is
reported unless both are true.

| Model | Method | Algorithm | Runnable now | Note |
|---|---|---:|---:|---|
| LLaVA-OV-7B | Base | yes | no | checkpoint adapter and GPU validation pending |
| LLaVA-OV-7B | TCD | yes | no | needs synchronized token-level dual branch |
| LLaVA-OV-7B | DINO-HEAL | unverified | no | upstream CLIP patch is not API-compatible |
| LLaVA-OV-7B | SEASON | yes | no | needs per-vision-layer homogenization hooks |
| Qwen2.5-VL-7B | Base | yes | no | checkpoint adapter and GPU validation pending |
| Qwen2.5-VL-7B | TCD | yes | no | needs synchronized token-level dual branch |
| Qwen2.5-VL-7B | DINO-HEAL | unverified | no | Qwen vision grid/fusion differs from CLIP |
| Qwen2.5-VL-7B | SEASON | yes | no | needs Qwen vision and decoder attention hooks |
| LLaVA-Video-7B | Base | yes | no | checkpoint adapter and GPU validation pending |
| LLaVA-Video-7B | TCD | yes | no | needs synchronized token-level dual branch |
| LLaVA-Video-7B | DINO-HEAL | conditional | no | only if CLIP tower satisfies upstream contract |
| LLaVA-Video-7B | SEASON | yes | no | needs per-vision-layer homogenization hooks |

The method-level numerical implementation and contracts are tested. No full
7B checkpoint was downloaded or claimed as validated during project setup.
