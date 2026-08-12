# Upstream Inventory

## VideoHallucer

The repository contains five released paired-QA datasets, model wrappers, and
`evaluations/evaluation_utils.py`. For every row it asks a basic question and a
hallucination question; official `accuracy` is the conjunction of both hits.
See `task_mapping.csv` for the non-inferred acronym mapping.

## EventHallusion

The repository contains question JSON for `entire`, `misleading`, and `mix`,
GPT-4o and Video-LLaVA inference, binary evaluator, GPT description judge, and
description evaluator. It does not contain a complete reusable TCD generation
implementation; TCD is reimplemented from paper arXiv:2409.16597.

## VidHalluc / DINO-HEAL

The clone contains BQA/MCQ/STH/TSH inference and evaluators plus a modified
Video-LLaVA `clip_encoder.py`. DINOv2 last-layer CLS-to-patch attention is
resized to the CLIP grid and fused as 0.3 standardized visual feature + 0.7
saliency. The vision encoder and DINOv2 are frozen.
