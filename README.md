# Reproducible VideoLLM Hallucination Benchmark

This project evaluates Base, TCD, DINO-HEAL, and a paper-faithful SEASON
reimplementation under one deterministic eight-frame protocol. Upstream
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
python3 scripts/aggregate_results.py --input results/metrics
```

Large checkpoints are deliberately not downloaded by setup scripts. Configure
local paths in `configs/models.yaml`, inspect the dry run, and only then run a
real benchmark.

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
[season_implementation.md](docs/season_implementation.md), and
[compatibility_matrix.md](docs/compatibility_matrix.md) before interpreting a
result table.
