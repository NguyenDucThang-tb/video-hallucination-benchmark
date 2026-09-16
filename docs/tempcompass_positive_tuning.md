# TempCompass positive-feature tuning

This tuning protocol keeps the controlled 8-frame TempCompass protocol and
uses the existing 109-point positive-feature grid. It selects a fixed random
one-third of the resolved instructions for each of:

- `multi-choice`: 527 of 1,580
- `yes_no`: 818 of 2,453
- `caption_matching`: 501 of 1,503

The subset is created once and reused by every model and grid point.

Create the subset on Gadi:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/create_tempcompass_tuning_subset.py \
  --seed 42 \
  --fraction 0.3333333333333333 \
  --output manifests/tempcompass_positive_tuning_seed42.json
```

Preview one model's 109-point plan:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_positive_feature_grid.py \
  --model qwen2.5-vl-7b \
  --benchmark tempcompass \
  --tasks multi-choice yes_no caption_matching \
  --subset-manifest manifests/tempcompass_positive_tuning_seed42.json \
  --prefix tempcompass_positive_grid_qwen \
  --no-resume
```

Each H200 PBS job runs one point for 30 minutes. Jobs use `resume=True`, so a
timed-out point can be resubmitted with the same model and point prefix.

The grid CSV includes task-level scores for all three tuning tasks. After the
grid completes, `scripts/summarize_tempcompass_grid.py` writes accuracy,
coverage, and correct counts by fine-grained temporal aspect. Captioning is
intentionally excluded from tuning because the official score requires the
TempCompass LLM judge.
