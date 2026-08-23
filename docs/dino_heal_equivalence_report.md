# DINO-HEAL Equivalence Report

Status: `PARTIAL APPROXIMATION`

Primary sources:

- Paper: https://arxiv.org/abs/2412.03735
- Official source: https://github.com/CyL97/VidHalluc/tree/main/DINO-HEAL

Paper-faithful DINO-HEAL requires DINO attention aligned to every target visual
patch, including exact patch-grid interpolation and fusion location. It also
requires evaluation of normalization enabled/disabled and DINO variants with
and without registers.

Current adapter gaps:

- Qwen reduces DINO evidence to one scalar per frame and scales broad vision
  outputs. This is not patch-level alignment.
- LLaVA-OV pools model features and DINO evidence per frame before applying a
  projector hook. This is not the official patch-level fusion path.
- `fuse_saliency` always normalizes features.
- Only `facebook/dinov2-large` is configured; the register variant grid is absent.

The code rejects silent DINO load/hook failure, but successful execution does
not establish equivalence. These outputs may be labeled only
`DINO-HEAL approximation` and are excluded from paper-compatible tables.
