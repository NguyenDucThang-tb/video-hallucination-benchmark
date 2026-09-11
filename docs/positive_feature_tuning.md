# Positive-feature tuning

The tuning protocol uses one persisted random subset for every grid point:

- 100 VidHalluc TSH videos
- 50 VidHalluc MCQ videos
- 100 VideoHallucer TPH branch videos (50 complete basic/hallucination pairs)
- 50 EventHallusion videos, retaining every question for each selected video

Generate the manifest once on Gadi:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/create_tuning_subset.py \
  --seed 42 \
  --tsh-videos 100 \
  --mcq-videos 50 \
  --tph-videos 100 \
  --event-videos 50 \
  --output manifests/positive_feature_tuning_seed42.json
```

Do not regenerate this file between grid points. The manifest stores exact
sample IDs; TPH selection always retains complete pairs and the runner adjusts
the pair denominator to the selected subset. Videos that cannot be resolved on
disk are excluded before random selection.

Preview the 109-point grid without inference:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_positive_feature_grid.py \
  --model qwen2.5-vl-7b \
  --subset-manifest manifests/positive_feature_tuning_seed42.json
```

Run one grid point as a GPU smoke test:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_positive_feature_grid.py \
  --model qwen2.5-vl-7b \
  --subset-manifest manifests/positive_feature_tuning_seed42.json \
  --start-index 1 \
  --stop-index 1 \
  --execute
```

Smoke-test one representative from all five ablation groups and validate that
the hook diagnostics contain the requested coefficients:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_positive_feature_grid.py \
  --model qwen2.5-vl-7b \
  --subset-manifest manifests/positive_feature_tuning_seed42.json \
  --prefix qwen_positive_ablation_smoke \
  --smoke-ablations
```

Run the complete grid by removing the two index options. Add
`--include-baseline` to include `(alpha, alpha_s, beta) = (0, 0, 0)`.
Every point gets a unique experiment name and an individual log. The live
summary is written to:

```text
results/tables/<prefix>.grid.csv
```

Only complete runs are ranked. The CSV records failed runs and marks the best
and worst complete configurations.

An ordinary experiment can override method parameters without editing the
repository defaults:

```yaml
methods: [positive_feature]
method_configs:
  positive_feature:
    alpha: 0.2
    alpha_s: 0.1
    beta: 0.6
subset_manifest: manifests/positive_feature_tuning_seed42.json
```

## LLaVA-Video batch runs

The same 109-point grid can be run with the official LLaVA-Video-7B Qwen2
checkpoint on an H200. Submit one PBS job per point; the scheduler controls how
many jobs run concurrently:

```bash
cd /scratch/jp09/dd9648/video-hallucination-benchmark
mkdir -p logs

for POINT in $(seq 1 109); do
  P=$(printf '%03d' "$POINT")
  qsub -P hn98 -v POINT="$POINT" \
    -N "llvpf_p${P}" \
    -o "$PWD/logs/llvpf_p${P}.h200.out" \
    -e "$PWD/logs/llvpf_p${P}.h200.err" \
    pbs/llava_video_positive_grid_h200.pbs
done
```

Each point writes an independent `*.grid.csv` and log under the
`llava_video_positive_spatial_batch_v2_h200_pNNN` prefix. This runner records the
Positive Feature hook diagnostics and benchmark metrics. It does not claim
logit deltas because the current LLaVA-Video positive-feature adapter does not
yet expose the Qwen logit-diagnostics contract.

The H200 runner sets `saliency_device=cuda` and batches all eight sampled
frames in one BiRefNet forward pass. LLaVA-Video diagnostics include the
BiRefNet, input preparation, projector hook, enhancement, and generation wall
times, plus the actual spatial foreground-mask shape.

## LLaVA-OV batch runs

LLaVA-OV uses the same 109-point grid and fixed subset. Submit one PBS job per
point on an H200:

```bash
cd /scratch/jp09/dd9648/video-hallucination-benchmark
mkdir -p logs

for POINT in $(seq 1 109); do
  P=$(printf '%03d' "$POINT")
  qsub -P hn98 -v POINT="$POINT" \
    -N "llovpf_p${P}" \
    -o "$PWD/logs/llovpf_p${P}.h200.out" \
    -e "$PWD/logs/llovpf_p${P}.h200.err" \
    pbs/llava_ov_positive_grid_h200.pbs
done
```

The LLaVA-OV runner uses the native video input (`videos=[video]`), CUDA
BiRefNet with an eight-frame batch, and records next-token logit deltas for
each point. If the checkpoint is not already discoverable from the Hugging
Face cache, export `LLAVA_OV_MODEL_DIR` in the PBS script or pass it with
`qsub -v`.
