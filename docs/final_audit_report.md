# Final audit report

## 1. Audit identity

- Local baseline: branch `main`, commit
  `cad2f352c9da2fe7b1d3b96f54341241fb0d16b3`.
- Remote: `https://github.com/NguyenDucThang-tb/video-hallucination-benchmark.git`.
- Worktree was clean before the audit edits.
- VideoHallucer: `8b785d1680465911cd2ce80c9f652837c0ba2abd`.
- EventHallusion: `aa544c21c7cd93b4685423cb94f77ab441f754bc`.
- VidHalluc: `e753864f5c2500c38523f97992355e2352bf8732`.

## 2. Dataset/evaluator conclusions

- VideoHallucer snapshot and strict pair metric match upstream behavior. The
  local evaluator now keeps incomplete pairs in the denominator, deduplicates
  retry records and requires both branches to contain their expected answer.
- EventHallusion binary evaluation now follows the upstream leading yes/no
  parser and keeps parser failures/missing outputs in the denominator. `mix`
  resolves either `mix/` or `interleave/`; description reference is
  `event_info.unexpected`. Description judging remains N/A.
- VidHalluc BQA is now question-level and requires all expected clips. MCQ and
  TSH retain all emitted annotations. STH binary accuracy is no longer
  presented as the official score; official STH remains N/A until SimCSE
  location scoring is executed.
- Missing videos produce explicit failed prediction records instead of being
  silently skipped. Full video resolution is unverified because Gadi scratch
  is not mounted in this environment.

## 3. Method conclusions

- No method under `src/methods/` is an official copied implementation.
- TCD is a paper-based reimplementation. The complete source implementation
  was not confirmed in EventHallusion. Its core numerical/subset logic exists,
  but real-model correctness is unverified.
- DINO-HEAL is partial: the adapters use token norms and frame-level scaling,
  not final-layer head-averaged CLS-to-patch attention aligned patch-by-patch.
- SEASON is partial: temporal homogenization is input-level rather than after
  every vision layer, and exact decoder visual-token attention mapping is not
  demonstrated. The positive-feature module is a separate extension, not a
  SEASON paper component.
- All model-method pairs are `Runnable now = no`; none has all eight required
  adapter/contract/test/smoke/GPU/checkpoint/video validation conditions
  evidenced by committed artifacts in this checkout.

## 4. Corrected defects

1. Corrected BQA evaluation unit and incomplete-question handling.
2. Removed misleading STH binary accuracy from the official metric column.
3. Kept missing/error records in metric denominators.
4. Corrected EventHallusion parser semantics and fixed split reporting.
5. Counted incomplete VideoHallucer pairs as incorrect.
6. Added latest-record deduplication with duplicate accounting.
7. Added explicit sampling/generation failure records and manifests.
8. Disabled unsupported compatibility claims.
9. Added a dataset-only validation command for Gadi.
10. Added fixed final tables with percentages, N/A and strict averages.
11. Added provenance, dataset and method-equivalence reports.
12. Added regression tests for the defects above.
13. Prevented metrics from mixing JSONL files from different experiment names.

## 5. Verification log

| Command | Result |
| --- | --- |
| `git diff --check` | PASS |
| `python3 -m compileall -q src scripts tests` | PASS |
| `python3 scripts/aggregate_results.py --input results/metrics --output-dir results/tables` | PASS; generated four N/A-only final artifacts |
| `python -m compileall ...` | NOT EXECUTED: `python` command is absent |
| `python3 -m pytest -q` | TESTS NOT EXECUTED: `pytest` is not installed |
| `python3 scripts/smoke_test.py` | TESTS NOT EXECUTED: `numpy` is not installed |
| `python3 scripts/run_benchmark.py --config configs/experiment1.yaml --dry-run` | NOT EXECUTED past import: `numpy` is not installed |

`ensurepip` is disabled in the system Python and there is no project `.venv`.
No package, checkpoint, dataset or paid service was downloaded during audit.

## 6. Remaining blockers

- Install the declared lightweight/test dependencies in a project environment
  and run the complete test suite.
- The reference image path under `/workspace/scratch/...png` is not available
  in this environment; the explicit table schema in the request was used.
- Run `scripts/validate_datasets.py` on Gadi with mounted real videos.
- Implement DINO CLS-to-patch attention and target patch-grid alignment.
- Implement SEASON vision-layer hooks and exact decoder attention mapping.
- Validate TCD branch state/KV caches on each real checkpoint.
- Commit artifact-backed smoke reports before enabling any compatibility pair.
- Run official STH SimCSE scoring and EventHallusion description judging where
  their dependencies/credentials are available.

## 7. Results table

`results/tables/final_results.{csv,md,tex,json}` contains the requested fixed
matrix. Every metric is N/A because no currently supported, fully validated
model-method result exists in this checkout. Existing external/Gadi numbers
were not imported because their provenance and corrected-evaluator status
cannot be established here. This is deliberate and is not a reproduced result.

**Chưa thể xác nhận từ môi trường hiện tại.**
