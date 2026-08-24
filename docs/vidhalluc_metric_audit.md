# VidHalluc metric audit

## TSH

The compatibility parser is a behavioral port of
`external/VidHalluc/eval/evaluation/eval_tsh.py`. Upstream uses the string
`"None"`; local records use Python `None`. Both are non-matching answers, not
Boolean `False` and not `BA`.

Official accuracy is `correct / total_entries`. Unparseable, empty and missing
outputs stay in the denominator. The evaluator separately reports parsed AB, BA,
A and B counts, parse coverage, empty outputs, missing predictions and runtime
failures.

The Gadi audit of existing LLaVA-OV Base TSH output found 600 records, 71 correct
and 529 incorrect under the upstream parser: `11.83%`. All 600 were accepted by
that parser. The earlier 491 unparseable outputs came from the old stricter local
parser. Parser acceptance does not imply correctness: A/B cannot equal AB/BA.

## STH

Official STH is not binary accuracy:

```text
classification_score = ((MCC + 1) / 2) ** 2
description_score = SimCSE location score using low threshold 0.5
overall = 0.6 * classification_score + 0.4 * description_score
```

The vendored evaluator treats every classification string other than `yes` as
the negative class for MCC. Local parsing keeps malformed output as `None`, then
the compatibility metric follows that policy while reporting unknown count.

Official STH score is unavailable without the specified SimCSE checkpoint.
Binary diagnostic accuracy must not be reported as official STH.

```bash
PYTHONPATH=. ./.venv/bin/python scripts/compare_vidhalluc_evaluators.py \
  --input results/raw --output-dir results/audit
```
