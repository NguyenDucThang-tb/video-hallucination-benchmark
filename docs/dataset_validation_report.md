# Dataset validation report

## Environment boundary

The configured video roots are under `/scratch/jp09/dd9648/`, which is not
mounted in the current local environment. Therefore video resolution counts
and real inference are **UNVERIFIED here**. Run `scripts/validate_datasets.py`
on Gadi to produce `results/dataset_validation.json`; this does not load a
checkpoint or require a GPU.

## Annotation inventory

Counts below were computed from the pinned annotation snapshots in this
checkout where available.

| Benchmark/task | Annotation unit | Units | Raw QA records | Official primary metric |
| --- | --- | ---: | ---: | --- |
| VideoHallucer ORH | pair | 200 | 400 | strict pair accuracy |
| VideoHallucer TPH | pair | 176 | 352 | strict pair accuracy |
| VideoHallucer SDH | pair | 200 | 400 | strict pair accuracy |
| VideoHallucer EFH | pair | 200 | 400 | strict pair accuracy |
| VideoHallucer ENFH | pair | 200 | 400 | strict pair accuracy |
| EventHallusion entire | video/question | 109 | 114 | binary question accuracy |
| EventHallusion misleading | video/question | 95 | 102 | binary question accuracy |
| EventHallusion mix | video/question | 193 | 193 | binary question accuracy |

VidHalluc annotation JSON files are not included in the local `external/`
snapshot and the configured Gadi root is unavailable, so current-environment
counts are N/A. The loader searches the configured video tree and supports an
annotation root one directory above it (the observed Gadi layout).

## Corrected behavior

- VidHalluc scoreable annotations are emitted even when their videos cannot be
  resolved. The runner records a `video_sampling` failure instead of skipping.
- BQA reports one unit per source question and requires every expected clip
  answer to be correct.
- VideoHallucer incomplete pairs remain in the denominator and are incorrect.
- EventHallusion accepts `mix` videos from either `mix/` or `interleave/`;
  `event_info.unexpected` is the description reference for this split.
- EventHallusion parser failures and missing outputs remain in the binary
  denominator. Description judging is N/A without the official judge/API.
- Duplicate prediction records are resolved by full sample identity, latest
  record wins, and duplicate counts are reported.

## Metric formulas

- VidHalluc BQA: question correct iff all clip-level answers are correct.
- VidHalluc MCQ and TSH: sample accuracy including missing/error records.
- VidHalluc STH: `0.6 * ((MCC + 1) / 2)^2 + 0.4 * SimCSE description score`.
  The local evaluator reports STH as N/A until the description component runs.
- VideoHallucer task: strict correct pairs divided by all annotated pairs.
- VideoHallucer AVG: strict macro-average over ORH/TPH/SDH/EFH/ENFH; N/A if
  any task is absent.
- EventHallusion AVG: total correct binary questions divided by all questions
  across configured splits (sample-weighted), matching the upstream evaluator.
