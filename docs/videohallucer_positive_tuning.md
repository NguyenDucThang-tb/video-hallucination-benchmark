# VideoHallucer TPH/SDH Positive Feature tuning

This experiment evaluates all 109 Positive Feature parameter points for
Qwen2.5-VL-7B on the full TPH and SDH tasks:

- `TPH`: 352 branch records, 176 complete pairs
- `SDH`: 400 branch records, 200 complete pairs

Each PBS job runs both tasks for one parameter point with a 30-minute
walltime. The grid summary reports strict pair accuracy separately for TPH and
SDH plus their unweighted mean.

Create the full manifest:

```bash
PYTHONPATH=. ./.venv/bin/python \
  scripts/create_videohallucer_tuning_manifest.py
```

Submit all 109 points:

```bash
for POINT in $(seq 1 109); do
  P=$(printf '%03d' "$POINT")
  qsub -P hn98 -q gpuhopper \
    -l walltime=00:30:00 \
    -v POINT="$POINT" \
    -N "vhr_grid_qwen_p${P}" \
    -o "$PWD/logs/videohallucer_positive_grid_qwen2_5_vl_7b_p${P}.out" \
    -e "$PWD/logs/videohallucer_positive_grid_qwen2_5_vl_7b_p${P}.err" \
    pbs/videohallucer_positive_grid_h200.pbs
done
```

View the task-level result table:

```bash
PYTHONPATH=. ./.venv/bin/python \
  scripts/summarize_videohallucer_grid.py
```
