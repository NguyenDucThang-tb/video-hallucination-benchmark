# Reproducible VideoLLM Hallucination Benchmark

This project is an audit-oriented harness for Base, TCD, DINO-HEAL, and
SEASON under one deterministic eight-frame protocol. TCD is a paper-based
reimplementation; the current DINO-HEAL and SEASON adapters are partial and
disabled for research tables. Upstream
benchmark code snapshots are vendored in `external/` so a plain `git clone`
contains the code needed to run on another machine. Large datasets, videos,
and model checkpoints are still kept outside git.

No benchmark number in this repository is synthetic. Missing checkpoints,
datasets, API credentials, or unsupported model-method combinations produce
an explicit `N/A`/error record.

## Quick start

```bash
python3 -m pip install -e '.[video,test]'
bash scripts/prepare_data.sh
python3 scripts/smoke_test.py
python3 scripts/run_benchmark.py --config configs/experiment1.yaml --dry-run
python3 scripts/run_benchmark.py --config configs/experiment1.yaml --smoke-test
python3 scripts/evaluate_results.py --input results/raw
python3 scripts/aggregate_results.py --input results/metrics/metrics.json
```

Large checkpoints are deliberately not downloaded by setup scripts. Configure
local paths in `configs/models.yaml`, inspect the dry run, and only then run a
real benchmark.

TempCompass is available through `configs/tempcompass_8frame.yaml`. Its four
official prompt formats are preserved; see
[`docs/tempcompass_protocol.md`](docs/tempcompass_protocol.md) for data layout
and the required second-stage LLM judging for caption generation.

Qwen2.5-VL 32B and 72B use the staged H200 launcher documented in
[`docs/qwen25_vl_large_benchmark.md`](docs/qwen25_vl_large_benchmark.md).
The full large-model matrix contains 220 resumable jobs, so run its mandatory
smoke stage before submitting one model/benchmark group at a time.

If you need to refresh the vendored upstream snapshots, use
`scripts/clone_repositories.sh` in a temporary checkout and update
`external/COMMITS.lock` with the exact commits used.

## Protocol guarantees

- Exactly 8 deterministic uniform samples per video.
- Short videos repeat deterministic indices; the policy is recorded.
- Base and every method consume the same manifest indices.
- TCD negatives are chronological subsets of those same 8 frames.
- Greedy decoding (`do_sample=false`, `temperature=0`).
- Raw output is retained alongside normalized output and parser state.
- Valid predictions are resumed, not overwritten.

Read [experimental_protocol.md](docs/experimental_protocol.md),
[method_equivalence_report.md](docs/method_equivalence_report.md), and
[compatibility_matrix.md](docs/compatibility_matrix.md) before interpreting a
result table.
