# Stage 1 Report

## Completed

- Cloned and pinned VideoHallucer, EventHallusion, and VidHalluc under
  `external/` without modifying upstream files.
- Read benchmark READMEs, inference/evaluator paths, DINO-HEAL patch, SEASON
  source TeX, and EventHallusion/TCD source TeX.
- Documented protocol, task mapping, compatibility, SEASON equations, and
  known evaluator issues.
- Implemented deterministic 8-frame sampling, manifest schema, auditable JSONL
  records, strict parsers, resume keys, benchmark loaders/evaluators, TCD,
  DINO-HEAL feature fusion, and SEASON numerical/core contracts.
- Added dry-run, smoke, evaluation, aggregation, and pinned clone scripts.
- Passed 19 unit tests and a non-research pipeline smoke test.

## Blockers to full benchmark execution

- This host has no `nvidia-smi`, no local Hugging Face 7B checkpoints, and no
  Hugging Face model cache.
- Video files for EventHallusion and the full VidHalluc data roots are not
  configured.
- The three model-specific token/attention/vision-hook adapters have not been
  GPU validated; dry-run therefore reports N/A rather than claiming support.
- EventHallusion description judging additionally needs `OPENAI_API_KEY`.
- TempCompass, TVBench, VideoMME, and MVBench upstream data/evaluator packages
  were not among the three repositories authorized for cloning in Stage 1.

No benchmark accuracy was generated or reported.
