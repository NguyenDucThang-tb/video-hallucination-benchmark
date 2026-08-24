# VidHalluc TSH/STH pipeline audit

Audit date: 2026-08-25

Local repository baseline for the expanded audit: `bfde71a927e954e6b29be0aa2686bd8ab3271cd8`.
The working tree was clean when the audit started.

Upstream reference: <https://github.com/CyL97/VidHalluc>. The vendored files under
`external/VidHalluc` match the locked TSH/STH evaluator and inference files at
`e753864f5c2500c38523f97992355e2352bf8732`. The code revision is pinned in
`external/COMMITS.lock`; the separately downloaded dataset revision remains unverified.

## Findings

### TSH

- The run line `base / vidhalluc / tsh` is a Base run. It is not evidence that TCD ran.
- The old local prompt was not the public VidHalluc prompt. It required only `AB` or
  `BA`, while upstream also asks the model to return a single action when only one is
  detected. The loader now uses the upstream instruction verbatim.
- The old local parser was not the official parser. It accepted punctuation such as
  `AB.`, which the public evaluator returns as `None`, and supported a different set
  of natural-language forms. The official parser is now ported separately from the
  permissive diagnostic parser.
- Upstream accuracy uses every annotation as the denominator. An unparseable answer
  remains `None` at record level but contributes no correct match in the official
  numerator. Metrics now report `official_accuracy`, `all_sample_accuracy`,
  `valid_only_accuracy`, `parse_coverage`, and the unparseable count separately.
- The Gadi reparse of all 600 existing Base outputs with the compatibility parser found
  71 correct and 529 incorrect, yielding official accuracy `71/600 = 11.83%` and parser
  coverage `600/600`. The old local parser rejected 491 of those outputs; this was
  parser disagreement, not 491 runtime failures.
- AB/BA ground-truth semantics are preserved; no label inversion was found in the
  loader. Tests cover all four correct/incorrect combinations through direct equality.

### STH

- The local prompt differed slightly in wording and now matches the public inference
  prompt.
- Public STH uses `princeton-nlp/sup-simcse-roberta-large`, CLS embeddings, cosine
  similarity, a low threshold of `0.5` in the actual `main()` call, transformed MCC,
  and `0.6 * classification_score + 0.4 * description_accuracy`.
- Binary scene-change accuracy is diagnostic only. It is not the official STH score.
- Without the SimCSE checkpoint, `official_accuracy`, `description_accuracy`, and
  `overall_score` remain `N/A`; the strict VidHalluc AVG must also remain `N/A`.
- The audit command can execute SimCSE only when an explicit local checkpoint is
  supplied. It loads the model once and batches unique scene strings.

### Dataset and video mapping

- TSH/STH now resolve the exact annotation filename/stem instead of using the broad
  fuzzy alias logic used by other local tasks.
- Invalid TSH/STH labels now raise an explicit error instead of being silently skipped.
- Missing videos continue to produce explicit failure records and remain in the
  denominator.
- Annotation counts, duplicate IDs, missing videos, and mapping correctness were not
  executable locally because `/scratch/jp09/dd9648/datasets_video_hallu/vidhalluc`
  is available only on Gadi. Run the audit command below there.

### Frame sampling

- Public VidHalluc first samples approximately one frame per second using rounded FPS,
  then uniformly caps the sequence at 32 frames. It does not pad short videos to 32.
- The previous local protocol used exactly 8 uniform frames over the whole video.
  This is a `FRAME PROTOCOL MISMATCH` for TSH/STH.
- TSH/STH now use the public strategy through task-specific configuration. The actual
  indices are preserved in every prediction record and manifest.
- The local decoder remains a robust full OpenCV decode, while upstream combines
  Decord metadata with an OpenCV frame count. The selection rule is matched, but reader
  equivalence is only partially verified and must be checked from Gadi manifests.
- Local SEASON currently requires exactly eight frames. It is therefore incompatible
  with the official 32-frame VidHalluc protocol until SEASON is adapted and validated;
  the runner will fail explicitly rather than silently switch protocols.

### Model input and generation

- LLaVA-OV and Qwen adapters use `apply_chat_template(..., add_generation_prompt=True)`
  and pass image tensors to the processor/model. New records capture the rendered
  prompt, input keys/shapes, frame count, and whether a vision tensor was supplied.
- The local LLaVA-OV adapter represents sampled video frames as multiple image inputs,
  while the public template uses the original LLaVA video stack and `modalities="video"`.
  This is not architecture-equivalent and prevents a full reproduction claim.
- Local generation is greedy with `do_sample=False`, `temperature=None` at model call,
  `num_beams=1`, cache enabled, and configured `max_new_tokens` (currently 128). Public
  inference uses 1024 tokens and `top_p=0.1` despite deterministic decoding. TSH likely
  does not need 1024 tokens, but the configuration is not identical.
- Raw generated text is stored before parsing. Runtime failures retain empty raw output,
  `is_correct=None`, an error, and a failure stage.

## Validation status

```text
TSH_CONFIG: MISMATCH
TSH_PROMPT: VERIFIED
TSH_PARSER: VERIFIED
STH_CONFIG: MISMATCH
STH_OFFICIAL_METRIC: SIMCSE_NOT_AVAILABLE
OVERALL: PARTIALLY CORRECT
```

The config mismatch classification remains because dataset revision, model architecture,
chat-template equivalence, and a real Gadi run with the official frame protocol have not
all been validated.

Current results are local evaluation results and are not yet verified against the
official VidHalluc evaluation protocol.

## Gadi audit command

This command only reads existing JSONL outputs; it does not run inference:

```bash
cd /scratch/jp09/dd9648/video-hallucination-benchmark
source .venv/bin/activate

PYTHONPATH=. ./.venv/bin/python scripts/audit_vidhalluc_tsh_sth.py \
  --raw-dir results/raw \
  --output-dir results/audit
```

To execute official STH description scoring, pass a local copy of the official model:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/audit_vidhalluc_tsh_sth.py \
  --raw-dir results/raw \
  --output-dir results/audit \
  --simcse-model /scratch/jp09/dd9648/huggingface/sup-simcse-roberta-large
```

After inspecting the audit artifacts, run a 5-10 sample controlled smoke test before a
new full run. Legacy records cannot contain rendered chat prompts or vision-input shapes;
those fields are captured only by runs made after this patch.

## Test status

- `python3 -m compileall -q src scripts tests`: passed.
- Official TSH parser parity check against vendored upstream: passed for 15 representative cases.
- `python -m pytest -q`: `TEST NOT EXECUTED`.
- Reason: the local execution environment has no `pytest` or project dependencies installed.
- GPU smoke test: `TEST NOT EXECUTED`.
- Reason: the local workspace has no VidHalluc dataset, model checkpoint, or GPU runtime.
