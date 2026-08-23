# VideoHallucer Base Reproducibility Audit

Audit time: 2026-08-23T16:08:46+07:00  
Local repository commit: `78380721f7e1659a5abbddd40529441440dedd82`  
Official VideoHallucer `main`: `8b785d1680465911cd2ce80c9f652837c0ba2abd`  
Official commit date: 2025-12-16T22:13:27+08:00  
Paper: [arXiv:2406.16338v1](https://arxiv.org/abs/2406.16338)  
Official repository: [patrick-tssn/VideoHallucer](https://github.com/patrick-tssn/VideoHallucer)

## 1. Gate Decision

**Status: NOT REPRODUCED**

```text
BASELINE NOT REPRODUCED.
TCD/DINO-HEAL COMPARISON IS NOT VALID YET.
```

The bundled annotation JSON files are byte-identical to current upstream, but
neither relevant paper protocol is fully established. The original VideoHallucer
paper does not evaluate LLaVA-OV-7B, Qwen2.5-VL-7B, or LLaVA-Video-7B and uses
baseline-specific defaults. SEASON Table 1 does use those backbones and states a
common eight-frame setup, but does not publish annotation hashes, exact frame
indices, processor revisions or checkpoint revisions. The VideoHallucer paper's
original Temporal split also differs from current upstream data. No Base raw predictions, videos, GPU environment, or checkpoints
are present in this local workspace, so the Gadi Base runs cannot be audited
record by record here.

The gate therefore stops before any TCD, DINO-HEAL or SEASON assessment.

## 2. Environment And Provenance

| Item | Observed value |
| --- | --- |
| Local Git commit | `78380721f7e1659a5abbddd40529441440dedd82` |
| Official VideoHallucer commit | `8b785d1680465911cd2ce80c9f652837c0ba2abd` |
| Python | 3.12.3 |
| PyTorch | unavailable in the local environment |
| Transformers | unavailable in the local environment |
| OpenCV | unavailable in the local environment |
| NumPy | unavailable in the local environment |
| Local `.venv` | absent |
| Base raw JSONL | absent from `results/raw` |
| Video files | absent from bundled `external/VideoHallucer` data |

`external/VideoHallucer` is a copied source tree, not an independent Git
checkout. Its annotation hashes and key evaluation source were compared against
a fresh checkout of the official commit above. The generated provenance table
is `results/audit/videohallucer_annotation_provenance.csv`.

## 3. Dataset And Pair Inventory

The bundled annotation JSON files are byte-identical to official upstream at
`8b785d1`. The actual Gadi dataset configured at
`/scratch/jp09/dd9648/datasets_video_hallu/videohallucer` is outside this
environment and remains unverified.

| Task | Raw annotation rows | Basic rows | Hallucination rows | Valid pairs | Unique videos | Missing videos locally | Missing branches | Duplicate pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ORH | 200 | 200 | 200 | 200 | 183 | 183 | 0 | 0 |
| TPH | 176 | 176 | 176 | 176 | 165 | 165 | 0 | 0 |
| SDH | 200 | 200 | 200 | 200 | 400 | 400 | 0 | 0 |
| EFH | 200 | 200 | 200 | 200 | 200 | 200 | 0 | 0 |
| ENFH | 200 | 200 | 200 | 200 | 200 | 200 | 0 | 0 |

Every annotation row has exactly one `basic` and one `hallucination` branch.
The loader maps them to a shared index-based `pair_id` such as `orh:0`, and to
sample IDs `orh:0:basic` and `orh:0:hallucination`. Pairing is independent of
prediction record order. It does depend on annotation row order, so annotation
hash provenance is required when reusing outputs.

ORH contains two repeated basic branch contents at annotation indices 117/148
and 119/138, but no duplicated complete pair. This is reported separately as
`duplicate_branch_rows=2`; it is not treated as a duplicate pair.

There is a version conflict that prevents VideoHallucer-paper-v1 reproduction:

- Paper v1 reports 200 pairs / 400 branch questions for TPH.
- Current upstream JSON has 176 pairs / 352 branch questions.
- Current upstream README says 376 Temporal questions after duplicate removal,
  which does not match the checked-in JSON count of 352 branch questions.

Therefore the current dataset matches current upstream source, but it does not
match the paper v1 Temporal protocol.

SEASON Table 1 does not identify its VideoHallucer annotation revision. Because
it was submitted after the upstream duplicate-removal note, current-upstream use
is plausible but remains an inference, not evidence.

```text
DATASET MISMATCH (paper v1 versus current upstream Temporal split)
```

The complete row-level inventory is in
`results/audit/videohallucer_pair_inventory.csv`.

## 4. Prompt Comparison

The paper and official evaluator append exactly:

```text
Answer the question using 'yes' or 'no'.
```

The local loader appends the same text after a newline. The question text comes
directly from the same annotation JSON.

| Item | Official upstream | Local | Same? | Evidence |
| --- | --- | --- | --- | --- |
| Question text | annotation `question` | annotation `question` | Yes | byte-identical JSON |
| Yes/no instruction | exact suffix above | exact suffix above | Yes | loader and upstream evaluator |
| System prompt | model-specific wrapper/default | model processor chat template | Unverified | different model families/wrappers |
| Chat template | baseline-specific | Hugging Face checkpoint template | Unverified | LLaVA-OV/Qwen adapters |
| Image/video placeholder | baseline-specific | eight image entries | No exact paper match | local adapters |
| Max output tokens | model default | 128 | Not generally the same | paper Appendix A.3.1 |

The question-level prompt is reproduced. The final tokenized conversation is
not established as paper-equivalent because the local models were not among the
paper baselines and use their own chat templates.

## 5. Frame Protocol Comparison

| Item | Paper | Official upstream | Local | Same? |
| --- | --- | --- | --- | --- |
| Number of frames | default of each baseline | baseline-specific | exactly 8 | No protocol-wide equivalence |
| Sampling strategy | baseline-specific | baseline-specific | nearest uniform linspace | Unverified/different |
| Short-video policy | not standardized | baseline-specific | repeat nearest linspace | Unverified |
| Frame ordering | chronological | baseline-specific chronological sampling | chronological | Partly |
| Resize/crop | baseline-specific | model processor-specific | HF processor-specific | Unverified |

For example, original-VideoHallucer LLaVA-NeXT-Video uses 32 uniformly sampled frames, while
PLLaVA is initialized with 16 frames. The paper explicitly says it retains each
model's default `num_of_frames`. A universal 8-frame protocol is a valid local
controlled experiment and matches SEASON's stated frame count, but exact SEASON
sampling indices, short-video behavior and preprocessing are not reported.

```text
PROTOCOL MISMATCH
```

Paper-compatible results and local 8-frame results must be stored and labeled
separately.

## 6. Checkpoint Comparison

| Model | Paper checkpoint | Local checkpoint | Processor | Dtype | Quantization | Same? |
| --- | --- | --- | --- | --- | --- | --- |
| LLaVA-OV-7B | not evaluated | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | checkpoint AutoProcessor | bf16 on supported GPU, otherwise fp16 | none configured | No paper counterpart |
| Qwen2.5-VL-7B | not evaluated | `Qwen/Qwen2.5-VL-7B-Instruct` | checkpoint AutoProcessor | bf16 on supported GPU, otherwise fp16 | none configured | No paper counterpart |
| LLaVA-Video-7B | not evaluated | `lmms-lab/LLaVA-Video-7B-Qwen2` | adapter not implemented in runnable path | unverified | unverified | No paper counterpart |

The paper was submitted in June 2024 and evaluates older model families listed
in Appendix A.3.1. These local Base rows may be new-model benchmark results, but
they are not exact reproductions of a paper row.

```text
CHECKPOINT MISMATCH
```

Exact local checkpoint snapshot revisions and processor/tokenizer revisions on
Gadi are not recorded in the available local raw artifacts.

## 7. Generation Configuration

| Parameter | Paper | Official upstream | Local | Same? |
| --- | --- | --- | --- | --- |
| `do_sample` | model default | baseline-specific | `false` | Not generally |
| `temperature` | model default | baseline-specific | disabled/greedy | Not generally |
| `num_beams` | model default | baseline-specific | 1 | Unverified |
| `max_new_tokens` | model default | baseline-specific | 128 | Not generally |
| `use_cache` | model default | usually true | true | Likely, not sufficient |
| EOS/stop | baseline-specific | wrapper stop criteria | HF model EOS | Unverified |

Examples in official wrappers include sampled decoding at temperature 0.1 or
0.2 and up to 1024 new tokens. The local controlled greedy protocol is therefore
not generally paper-compatible.

## 8. Parser Comparison

Official VideoHallucer scoring checks whether the expected answer token occurs
anywhere in raw output using a case-insensitive word-boundary regex. The local
VideoHallucer evaluator implements the same check directly in `official_hit`.
The generic local parser is stricter: it returns `ambiguous` if both `yes` and
`no` occur. Importantly, the current VideoHallucer pair evaluator ignores that
strict parsed value and re-scores raw output with official-compatible matching.

| Raw output | Official parser for expected answer | Strict local parser | Same? |
| --- | --- | --- | --- |
| `yes` | yes | yes | Yes |
| `no` | no | no | Yes |
| `Yes.` | yes | yes | Yes |
| `No.` | no | no | Yes |
| `The answer is yes.` | yes | yes | Yes |
| `The answer is no.` | no | no | Yes |
| `Yes, but maybe no.` | matches either expected token | ambiguous | No |
| `No, although yes.` | matches either expected token | ambiguous | No |

The audit emits both `official_compatible_strict_pair_accuracy` and
`strict_local_strict_pair_accuracy`; neither overwrites the other.

## 9. Metric And Denominator

The paper and official evaluator define a correct pair as:

```text
A pair is correct only when both the basic question
and the hallucination question are correct.
```

Local `_pair_stats` follows this rule. New records carry
`expected_task_pairs` from the annotation loader, so incomplete or entirely
absent pairs remain in the denominator. Legacy records without this metadata
are labeled `UNVERIFIED_DENOMINATOR`; their observed-only diagnostic score is
not exposed as research `accuracy`.

The independent audit also uses the annotation inventory as denominator and
reports missing records explicitly.

Reported audit metrics are:

- `branch_accuracy`
- `official_compatible_strict_pair_accuracy`
- `strict_local_strict_pair_accuracy`
- `local_current_pair_accuracy`

## 10. AVG Comparison

The repository's published leaderboard uses the arithmetic mean of the five
task-level strict pair accuracies. The local `strict_macro_average` implements
that formula and requires all five tasks.

The audit also reports a pooled pair-weighted average. These differ now because
TPH has 176 pairs while the other tasks have 200.

- Official/reporting AVG: macro-average across ORH, TPH, SDH, EFH, ENFH.
- Diagnostic pooled AVG: total correct pairs divided by total expected pairs.

They must not be conflated.

## 11. Existing Base Results

No Base raw prediction JSONL exists in this local checkout. Consequently:

- `results/audit/base_record_audit.csv` contains its schema but no fabricated rows.
- `results/audit/base_metric_comparison.csv` contains its schema but no fabricated scores.
- The previously displayed Gadi summary values cannot be validated from this
  environment because their underlying raw files, manifests, checkpoint
  metadata, and installed runtime are unavailable here.

```text
Chua the xac nhan tu moi truong hien tai.
```

Run the audit script on Gadi with the actual dataset and `results/raw` to fill
these files. No GPU is required for that audit:

```bash
cd /scratch/jp09/dd9648/video-hallucination-benchmark
source .venv/bin/activate

PYTHONPATH=. ./.venv/bin/python scripts/audit_videohallucer_base.py \
  --dataset-root /scratch/jp09/dd9648/datasets_video_hallu/videohallucer \
  --upstream-root external/VideoHallucer/videohallucer_datasets \
  --raw-dir results/raw \
  --output-dir results/audit
```

## 12. Verified And Unverified Causes Of Difference

Verified causes that prevent a paper reproduction claim:

1. The target local model checkpoints are absent from the paper's model set.
2. Local inference fixes all models to eight frames; paper inference retains
   each model's default frame count.
3. Local inference fixes greedy decoding and 128 output tokens; paper/upstream
   baseline generation is model-specific.
4. The current TPH annotation has 176 pairs, while paper v1 reports 200.

Still unverified from this environment:

1. Whether Gadi videos exactly match current upstream files.
2. Whether any Gadi videos are missing or undecodable.
3. Exact checkpoint, processor, tokenizer, and chat-template revisions used by
   each completed Base run.
4. Whether every expected Base record and both pair branches are present.
5. The difference between official-compatible and strict-local parsing on the
   actual raw outputs.

## 13. Final Answers

| Question | Answer |
| --- | --- |
| Base local reproduced the paper? | **No: NOT REPRODUCED.** |
| Dataset correct? | Annotation matches current upstream; Gadi videos are unverified; current TPH differs from paper v1. |
| Prompt correct? | Question suffix matches; complete model chat prompt is not paper-equivalent/verified. |
| Frame sampling correct? | Correct for the declared local 8-frame protocol, not for paper reproduction. |
| Checkpoint correct? | No exact paper counterpart for the three target models. |
| Parser correct? | VideoHallucer evaluator is official-compatible; strict generic parser differs on ambiguous responses. |
| Metric correct? | Pair rule is correct; denominator can omit completely absent pairs and must be audited against annotations. |
| AVG correct? | Macro formula is correct when all five complete task metrics are present. |
| Main source of Base difference? | Verified protocol/checkpoint/dataset-version mismatches; Gadi record-level causes remain unverified. |
| May TCD/DINO-HEAL be concluded now? | **No. Baseline Reproducibility Gate has not passed.** |
