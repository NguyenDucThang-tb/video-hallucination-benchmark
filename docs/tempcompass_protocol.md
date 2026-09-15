# TempCompass Protocol

## Source and scope

The vendored snapshot is commit `e1b463166400633e6061962d890a9ae85db29f70`
from the official TempCompass repository. It contains 7,540 instructions over
410 videos:

| Task | Instructions |
|---|---:|
| Multi-Choice QA | 1,580 |
| Yes/No QA | 2,453 |
| Caption Matching | 1,503 |
| Caption Generation | 2,004 |

The benchmark covers action, direction, speed, event order, and attribute
change, with fine-grained aspect labels copied from `meta_info.json`.

## Inference protocol

`configs/tempcompass_8frame.yaml` runs all three project models and all five
methods. Every run uses 8 deterministic uniform frames and greedy decoding.
This is the repository's controlled comparison protocol; it preserves the
official task instructions but does not claim to reproduce every model-specific
frame policy from the paper.

The official answer suffixes from the 2024-03-23 upstream revision are used
verbatim:

- Multi-choice and caption matching: `Please directly give the best option:`
- Yes/no: `Please answer yes or no:`
- Caption generation: the question already ends in `Generated Caption:`

## Evaluation protocol

The multi-choice, yes/no, and caption-matching parsers port the upstream
hand-written rules. `rule_only_accuracy` treats unmatched responses as wrong,
which is equivalent to upstream `--disable_llm`.

The paper's official pipeline sends unmatched responses to an LLM judge, and
caption generation always requires that judge. Therefore `official_accuracy`
is emitted only when no records require judging. Captioning records retain the
raw response and use `is_correct: null`; the harness never invents a score.

## Data layout

The default Gadi layout is:

```text
/scratch/jp09/dd9648/datasets_video_hallu/tempcompass/
└── videos/ (an extra archive directory level is also accepted)
    ├── 1034419625.mp4
    ├── 1034419625_reverse.mp4
    └── ...
```

Validate it before a run:

```bash
python3 scripts/verify_tempcompass_data.py \
  --video-root /scratch/jp09/dd9648/datasets_video_hallu/tempcompass
```

The expected count is 410.

For `gdown>=6.2`, pass the Google Drive file ID directly; the removed
`--fuzzy` option must not be used.

After inference, convert a raw JSONL file back to the nested schema accepted by
the vendored official evaluators:

```bash
python3 scripts/export_tempcompass_predictions.py \
  --input results/raw/RUN__MODEL__METHOD__tempcompass__TASK.jsonl \
  --output external/TempCompass/predictions/RUN/TASK.json
```
