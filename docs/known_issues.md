# Known Issues and Upstream Deviations

## EventHallusion mix evaluator

`external/EventHallusion/gpt4o_judge.py` checks `split == 'interleave'`, while
the released dataset and inference use `mix`. Consequently, upstream routes
mix descriptions to the misleading template and `event_info.caption`. Our
separate evaluator maps `mix` to `event_info.unexpected`. Upstream is retained
unchanged. Reports must show both evaluator variants when GPT judging runs.
Local data releases may additionally store the `mix` videos under an
`interleave/` directory; the benchmark loader accepts both layouts.

## EventHallusion API

Description accuracy relies on GPT-4o judging. The pipeline never calls it
without `OPENAI_API_KEY`, never embeds a key, and retains raw judgement and
parser errors. Binary evaluation does not require an API.

## VideoHallucer parser

The upstream regex searches expected yes/no anywhere in the full answer. This
can count contradictory text. Our parser requires exactly one unique yes/no
token and marks both-token responses ambiguous. Both definitions are recorded
when reproducing official numbers.

The README command contains `--eval_obj`, but `evaluation.py` defines
`--eval_obj_rel`. Dataset statistics and current JSON counts may differ due to
the October 2025 temporal duplicate removal.

## VidHalluc

Upstream inference defaults to 32 frames and resamples internally. This project
bypasses that sampler and enforces the shared 8-frame manifest. STH invokes a
large SimCSE model during evaluation; absence of its checkpoint is a blocker,
not a zero score.

## DINO-HEAL

The released code patches Video-LLaVA's CLIP tower. Its exception handler
silently returns CLIP-only features when DINOv2 fails. Our wrapper treats that
as an error and does not label the result DINO-HEAL. The patch is spatial
saliency enhancement, not a temporal reasoning module. Its default upstream
experiments use 32 frames; this protocol uses 8.

## Large-model status

No 7B checkpoint was automatically downloaded. Model-specific adapters require
GPU validation and are currently N/A in the runnable matrix. Pipeline smoke
tests use a clearly named deterministic smoke adapter and produce no reported
research metric.
