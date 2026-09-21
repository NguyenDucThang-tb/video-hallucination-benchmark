# Qwen2.5-VL 32B/72B benchmark protocol

This extension runs Qwen2.5-VL-32B-Instruct and Qwen2.5-VL-72B-Instruct on
all five registered video benchmarks and all five inference methods.

## Experiment matrix

- Models: `qwen2.5-vl-32b`, `qwen2.5-vl-72b`
- Methods: `base`, `tcd`, `dino_heal`, `season`, `positive_feature`
- Benchmarks: VidHalluc, VideoHallucer, EventHallusion, TempCompass,
  MotionBench
- Tasks: 22 in total
- Full matrix: 2 x 5 x 22 = 220 independently resumable PBS jobs

The dataset prompts, sampling, parsers, and evaluators are unchanged. The PBS
launcher caps Qwen visual input to `min_pixels=256*28*28` and
`max_pixels=384*28*28` for a controlled, feasible large-model run. MotionBench
keeps the repository protocol: 16 frames for all methods except the existing
8-frame SEASON setup.

## Method transfer policy

The large models have not been independently tuned. Positive Feature uses the
Qwen-7B selections as transfer configurations:

| Benchmark scope | alpha | alpha_s | beta |
| --- | ---: | ---: | ---: |
| VidHalluc, VideoHallucer, EventHallusion | 0.4 | 0.4 | 0.8 |
| TempCompass | 0.0 | 0.0 | 0.6 |
| MotionBench | 0.0 | 0.0 | 0.2 |

SEASON keeps the final-four-language-layer policy. Qwen-32B therefore uses
layers 60-63 and Qwen-72B uses layers 76-79. These are transferred policies,
not tuned results, and should be described as such in reported experiments.

## Download checkpoints on Gadi

```bash
cd /scratch/jp09/dd9648/video-hallucination-benchmark
source .venv/bin/activate

export HF_HOME=/scratch/jp09/dd9648/huggingface
export HF_HUB_CACHE="$HF_HOME"

huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct \
  --cache-dir "$HF_HOME"
huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct \
  --cache-dir "$HF_HOME"
```

## Mandatory smoke test

The launcher is dry-run by default. This first command only prints the 10 PBS
commands (2 models x 5 methods):

```bash
PYTHONPATH=. ./.venv/bin/python scripts/submit_qwen25_vl_large.py --smoke
```

Submit those smoke jobs explicitly:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/submit_qwen25_vl_large.py \
  --smoke --submit --run-tag qwen25_vl_large_smoke_v1
```

The smoke uses two EventHallusion `entire` samples. Do not submit the full
matrix until every model/method log exits with status 0 and no CPU/disk
offload error.

## Full runs

Submit in small stages to respect the Gadi per-project queue limit. For
example, one model and one benchmark at a time:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/submit_qwen25_vl_large.py \
  --models qwen2.5-vl-32b \
  --benchmarks eventhallusion \
  --submit --run-tag qwen25_vl_large_v1
```

The default resources are one H200 and 96 GB host RAM for 32B, and two H200s
and 192 GB host RAM for 72B. `device_map=auto` shards 72B across both GPUs.
The job fails immediately if Transformers places any weights on CPU or disk.

To inspect all 220 task states:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/status_qwen25_vl_large.py \
  --run-tag qwen25_vl_large_v1
```

All task jobs use stable prefixes and `resume: true`, so resubmitting the same
model/method/benchmark/task continues from valid JSONL records.
