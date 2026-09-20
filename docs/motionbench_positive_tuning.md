# MotionBench positive-feature tuning

This experiment tunes the 109 positive-feature parameter points independently
for Qwen2.5-VL-7B, LLaVA-OV-7B, and LLaVA-Video-7B on a fixed MotionBench dev
subset. It does not replace the positive-feature defaults used by other
benchmarks.

## Tuning subset

The subset is sampled once with seed 42. One fifth of the public dev examples
is selected independently from each task, using `ceil(task_size * 0.2)`:

| Task | Full dev | Tuning records |
|---|---:|---:|
| `action_order` | 519 | 104 |
| `location_related_motion` | 546 | 110 |
| `motion_recognition` | 1,478 | 296 |
| `motion_related_objects` | 690 | 138 |
| **Total** | **3,233** | **648** |

Every model and parameter point uses exactly the same sample IDs. The evaluator
denominators are rewritten to the subset counts, so each task accuracy and the
unweighted four-task mean are valid subset metrics.

Create the manifest on Gadi after the MotionBench data has been verified:

```bash
cd /scratch/jp09/dd9648/video-hallucination-benchmark
source .venv/bin/activate

PYTHONPATH=. ./.venv/bin/python scripts/create_motionbench_tuning_subset.py
```

## Grid protocol

- 109 positive-feature points from `positive_feature_grid()`.
- Four task accuracies and their unweighted arithmetic mean select the winner.
- 16 uniformly sampled frames for all three models.
- Greedy decoding with at most 16 new tokens.
- One H200 and 30 minutes per model/point job.
- Resume is enabled and each point has a stable experiment prefix.

The selected full-data MotionBench Positive Feature configurations are now
available automatically through the `motionbench` benchmark override:

| Model | Point | alpha | alpha_s | beta | Tuning mean |
|---|---:|---:|---:|---:|---:|
| LLaVA-OV-7B | 008 | 0.0 | 0.8 | 0.0 | 57.91% |
| LLaVA-Video-7B | 029 | 0.8 | 0.8 | 0.0 | 61.34% |
| Qwen2.5-VL-7B | 010 | 0.0 | 0.0 | 0.2 | 62.61% |

These overrides apply only when `benchmark: motionbench` is resolved. They do
not replace the model defaults used by the older benchmark suites.

Preview one point without inference:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_positive_feature_grid.py \
  --model qwen2.5-vl-7b \
  --benchmark motionbench \
  --subset-manifest manifests/motionbench_positive_tuning_seed42.json \
  --prefix motionbench_positive_grid_preview \
  --start-index 1 --stop-index 1
```

Submit in point ranges so the project queue limit is not exceeded. For example,
the first range submits 120 jobs (40 points times three models):

```bash
mkdir -p logs
START=1
STOP=40

for MODEL in qwen2.5-vl-7b llava-ov-7b llava-video-7b; do
  SAFE_MODEL="${MODEL//./_}"
  SAFE_MODEL="${SAFE_MODEL//-/_}"
  for POINT in $(seq "$START" "$STOP"); do
    P=$(printf '%03d' "$POINT")
    qsub -P hn98 -q gpuhopper \
      -v MODEL="$MODEL",POINT="$POINT" \
      -N "mb_grid_${SAFE_MODEL}_p${P}" \
      -o "$PWD/logs/mb_grid_${SAFE_MODEL}_p${P}.out" \
      -e "$PWD/logs/mb_grid_${SAFE_MODEL}_p${P}.err" \
      pbs/motionbench_positive_grid_h200.pbs
  done
done
```

After that range has left the queue, repeat with `START=41 STOP=80`, then
`START=81 STOP=109`. Existing jobs can reduce the available project slots, so
check `qstat -swu "$USER"` before each submission range.

The per-point CSV is written under `results/tables/`. Each row includes the
four task accuracies, `mean_score`, completion state, record counts, parameter
values, and rank.

Merge all completed points and print the best point for each model:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/summarize_motionbench_grid.py
```
