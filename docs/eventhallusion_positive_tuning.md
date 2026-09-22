# EventHallusion Positive Feature tuning

This experiment evaluates all 109 Positive Feature parameter points for
Qwen2.5-VL-7B. Every point runs all three EventHallusion tasks:

- `entire`: 114 QA records
- `misleading`: 102 QA records
- `mix`: 193 QA records

The full grid therefore has 109 PBS jobs, each with a 30-minute walltime. Each
point writes separate task metrics and an unweighted mean of the three task
accuracies to its grid CSV.

The manifest retains every annotation, including unresolved videos. The
current `mix` data has 40 unresolved records; those remain in the denominator
and are reported as `complete_with_errors` rather than silently dropping the
point from tuning.

Create the full manifest on Gadi:

```bash
PYTHONPATH=. ./.venv/bin/python \
  scripts/create_eventhallusion_tuning_manifest.py
```

Submit one H200 job per point:

```bash
for POINT in $(seq 1 109); do
  P=$(printf '%03d' "$POINT")
  qsub -P hn98 -q gpuhopper \
    -l walltime=00:30:00 \
    -v POINT="$POINT" \
    -N "eh_grid_qwen_p${P}" \
    -o "$PWD/logs/eventhallusion_positive_grid_qwen2_5_vl_7b_p${P}.out" \
    -e "$PWD/logs/eventhallusion_positive_grid_qwen2_5_vl_7b_p${P}.err" \
    pbs/eventhallusion_positive_grid_h200.pbs
done
```

Inspect all points and task-level percentages:

```bash
PYTHONPATH=. ./.venv/bin/python \
  scripts/summarize_eventhallusion_grid.py
```
