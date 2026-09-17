# MotionBench protocol

## Upstream snapshot

- Repository: <https://github.com/zai-org/MotionBench>
- Commit: `dcc9b0713c9b92d1c5d4ec7ee0b6dd51f86325a3`
- Paper: <https://arxiv.org/abs/2501.02955>
- Dataset: <https://huggingface.co/datasets/zai-org/MotionBench>
- Dataset revision used by the download command below:
  `f099db892172a015c489507c9abe56b036d960ef`

The vendored metadata contains 8,052 multiple-choice questions over 5,385 unique
videos. It has 4,018 public-label dev questions and 4,034 private-label test
questions. Local accuracy is therefore valid only for dev; test predictions must
be submitted to the official leaderboard.

## Evaluation dimensions

MotionBench is one multiple-choice benchmark with six fine-grained dimensions:

| Code | Category | Dev | Test | Purpose |
|---|---|---:|---:|---|
| MR | Motion Recognition | 1,478 | 1,466 | Recognize fine-grained actions and motion patterns. |
| LM | Location-related Motion | 546 | 597 | Understand direction, trajectory, and spatial changes. |
| CM | Camera Motion | 385 | 390 | Identify camera movement independently of object motion. |
| MO | Motion-related Objects | 690 | 725 | Associate motion with the correct object or actor. |
| AO | Action Order | 519 | 482 | Recover temporal ordering of actions. |
| RC | Repetition Count | 400 | 374 | Count repeated actions or motion cycles. |

The default internal task is `dev`; the evaluator reports all six dimensions.
`test` is supported for leaderboard export. The six category slugs can also be
run independently to parallelize dev inference:

`motion_recognition`, `location_related_motion`, `camera_motion`,
`motion_related_objects`, `action_order`, and `repetition_count`.

## Prompt and answer parsing

The loader preserves each official `question` string, including its answer
options, and appends only:

```text
Answer with only the option letter.
```

This constrains generation to the answer format expected by MotionBench. The
parser ports upstream `polish_answer`; only `A`, `B`, `C`, or `D` is accepted as
a parsed answer. Local scoring uses accuracy over the full expected dev
denominator, so parser and runtime failures count as incorrect.

## Frame protocol

The paper's model comparison uses model-dependent frame counts, so it does not
define one universal leaderboard frame count. This repository uses 16 uniformly
sampled frames for every model and method. This is a controlled comparison and
matches the 16-frame input used in the paper's temporal-equilibrium fusion
experiments, but it should not be described as reproducing every paper row.

The common experiment is in `configs/motionbench_16frame.yaml`. It evaluates:

- Models: LLaVA-OV-7B, Qwen2.5-VL-7B, LLaVA-Video-7B.
- Methods: base, TCD, DINO-HEAL, SEASON, positive feature.
- Generation: greedy decoding, 16 output tokens, no sampling.
- Sampling: exactly 16 uniform frames, with deterministic repetition for short videos.
- Resume: enabled for full runs.

SEASON receives a MotionBench-only `expected_frame_count: 16` override. Its
existing 8-frame setting remains unchanged for every other benchmark.

## Data verification and test export

```bash
PYTHONPATH=. ./.venv/bin/python scripts/verify_motionbench_data.py \
  --video-root /scratch/jp09/dd9648/datasets_video_hallu/motionbench
```

After a test run, export predictions in the upstream `{uid: letter}` format:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/export_motionbench_predictions.py \
  results/raw/EXPERIMENT__MODEL__METHOD__motionbench__test.jsonl \
  results/submissions/motionbench_MODEL_METHOD.json
```
