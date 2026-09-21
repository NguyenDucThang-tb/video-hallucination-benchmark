# EventHallusion Latency Protocol (NVIDIA H200)

This experiment compares three 7B video-language models and five inference
methods on all three EventHallusion splits.

## Controlled setup

- Models: `llava-ov-7b`, `qwen2.5-vl-7b`, `llava-video-7b`
- Methods: `base`, `tcd`, `dino_heal`, `season`, `positive_feature`
- Tasks: `entire` (114), `misleading` (102), `mix` (193)
- Hardware: one NVIDIA H200 80GB per PBS job
- Sampling: 8 uniform frames
- Batch size: 1
- Decoding: deterministic, one beam, at most 128 new tokens
- Repetitions: one complete run per model/method/task combination
- Timer boundary: `generate_batch`; model loading and video sampling are not timed
- Warm-up: exclude the first five successful samples in each job from the
  steady-state latency statistics

The run consists of 45 jobs (3 models x 5 methods x 3 tasks). Every job must
use a fresh prefix and `resume: false`; otherwise restarted processes introduce
additional warm-up periods into the same result file.

`mix` currently contains unresolved source videos. These samples are retained
as errors for coverage and accuracy, but are excluded from latency statistics
because no inference occurred.

Positive Feature uses the normal EventHallusion model-specific settings:

| Model | alpha | alpha_s | beta |
|---|---:|---:|---:|
| LLaVA-OV-7B | 0.8 | 0.1 | 0.6 |
| Qwen2.5-VL-7B | 0.4 | 0.4 | 0.8 |
| LLaVA-Video-7B | 0.2 | 0.1 | 0.0 |

## Submit

Use a new `RUN_TAG` for every clean measurement:

```bash
cd /scratch/jp09/dd9648/video-hallucination-benchmark
mkdir -p logs

RUN_TAG="eventhallusion_latency_h200_v1"

for MODEL in llava-ov-7b qwen2.5-vl-7b llava-video-7b; do
  SAFE_MODEL="${MODEL//./_}"
  SAFE_MODEL="${SAFE_MODEL//-/_}"

  for METHOD in base tcd dino_heal season positive_feature; do
    for TASK in entire misleading mix; do
      PREFIX="${RUN_TAG}_${SAFE_MODEL}_${METHOD}_${TASK}"

      qsub -P hn98 -q gpuhopper \
        -l walltime=02:00:00 \
        -v MODEL="$MODEL",METHOD="$METHOD",TASK="$TASK",PREFIX="$PREFIX" \
        -N "eh_lat_${SAFE_MODEL}_${METHOD}_${TASK}" \
        -o "$PWD/logs/${PREFIX}.out" \
        -e "$PWD/logs/${PREFIX}.err" \
        pbs/eventhallusion_latency_h200.pbs
    done
  done
done
```

If the project queue limit prevents all 45 jobs from being submitted together,
submit one model at a time without changing `RUN_TAG`.

## Monitor and summarize

```bash
qstat -swu dd9648

grep -H -E \
  'GPU:|START|END|LATENCY COMPLETE|Traceback|OutOfMemory|Exit Status' \
  logs/eventhallusion_latency_h200_v1_*.out \
  logs/eventhallusion_latency_h200_v1_*.err 2>/dev/null

PYTHONPATH=. ./.venv/bin/python scripts/summarize_eventhallusion_latency.py \
  --run-tag eventhallusion_latency_h200_v1 \
  --warmup 5
```

The summary is written to:

- `results/tables/eventhallusion_latency_h200_v1.latency.csv`
- `results/tables/eventhallusion_latency_h200_v1.latency.md`
