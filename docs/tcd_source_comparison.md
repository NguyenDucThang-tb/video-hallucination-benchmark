# TCD Source Comparison

Audit baseline: local commit `d7aea32` (2026-08-22). The working tree was clean
before this audit.

Primary sources:

- Paper: [EventHallusion, arXiv:2409.16597](https://arxiv.org/pdf/2409.16597)
- Upstream: [Stevetich/EventHallusion](https://github.com/Stevetich/EventHallusion)
- Upstream inference entry point:
  [run_inference.py](https://github.com/Stevetich/EventHallusion/blob/master/inference_template/videollava/eval/self_bench/run_inference.py)

The EventHallusion repository publishes benchmark evaluation and a standard
Video-LLaVA inference template. No complete TCD decoding implementation was
found in its public source tree. Therefore:

> The local TCD is a paper-based reimplementation; official upstream implementation was not confirmed.

| Item | Paper | Upstream code | Local code | Status | Evidence |
| ---- | ----- | ------------- | ---------- | ------ | -------- |
| Method | Training-free temporal contrastive decoding | No complete TCD path found | `TCDMethod` performs dual-branch autoregressive decoding | MATCH | Paper Sec. 4, Eqs. 1-5; `src/methods/tcd/tcd_method.py` |
| Original branch | `logit(y_t | V, x, y_<t)` | Standard single-video generation only | Original sampled frames, prompt, shared generated prefix | MATCH | Paper Eq. 1; `TCDMethod.generate` |
| Negative branch | Chronologically downsampled `S(V)` | Not implemented | Chronological subset of the already sampled input frames | MATCH | Paper Eq. 2; `chronological_downsample` |
| Negative source | Downsample original video | Not specified in code | Uses the same manifest frames; does not reopen the video | MATCH | `scripts/run_benchmark.py` samples once before method dispatch |
| Frame order | Chronological | Not implemented | Sorted uniform positions, no shuffle or reversal | MATCH | `chronological_downsample`; unit test |
| Frame counts | LLaVA-NeXT 32/8, VILA 12/8, VideoChat2 16/4 | Not implemented | Benchmark protocol 8/4 for all supported adapters | PARTIAL | Paper Sec. 5 implementation details; `configs/sampling.yaml`, `configs/methods.yaml` |
| Qwen negative processing | Not discussed | Not implemented | Fixed: processor receives exactly 4 negative frames; no second stride | MATCH | `Qwen25VLAdapter._apply_branch_transform`; regression test |
| Visual metadata | Must describe each branch input | Not implemented | Processor independently creates pixel tensors and grid metadata for each branch | MATCH | Qwen/LLaVA `prepare_branch` and `_prepare_inputs` |
| Prompt | Same `x` in both branches | Not implemented | Same prompt passed to both `prepare_branch` calls | MATCH | Paper Eqs. 1-2; `TCDMethod.generate` |
| Generated prefix | Same `y_<t` in both branches | Not implemented | Same `generated` list passed to both `decode_step` calls | MATCH | Paper Eqs. 1-2; SpyModel test |
| Token-level decoding | Logits contrasted at every autoregressive timestep | Not implemented | One original and one negative forward per output token | MATCH | Paper Sec. 4; `TCDMethod.generate` |
| Contrast formula | `(1+alpha) z_ori - alpha z_con` | Not implemented | Same formula | MATCH | Paper Eq. 3; `contrast_logits` |
| Threshold | `t=beta*max(z_ori)` in raw-logit space | Not implemented | Same threshold; values of mixed `z` below it become `-inf` | MATCH | Paper Eqs. 4-5; `contrast_logits` |
| Alpha/beta defaults | Ablation favors alpha 0.5; beta is robust around 0.5 | Not implemented | alpha 0.5, beta 0.5 | MATCH | Paper Sec. 5; `configs/methods.yaml` |
| All-masked behavior | Not specified | Not implemented | Explicit `original_argmax` fallback, counted in diagnostics; strict `error` mode available | PARTIAL | `TCDConfig.all_masked_behavior`; fallback tests |
| Token selection | Highest-probability token; greedy search | Standard template is greedy | `argmax` over masked logits | MATCH | Paper Secs. 4-5; `TCDMethod.generate` |
| EOS | Not detailed | Standard generation stopping criteria | Stops when adapter EOS token is selected | PARTIAL | Adapter `is_eos`; EOS unit test |
| Output decode | Not detailed | Batch-decodes generated sequence | Decodes the complete generated token sequence once | MATCH | Adapter `decode_token_ids`; SpyModel test |
| KV cache | Not specified | Base template sets `use_cache=True` | Independent state and `past_key_values` per branch | PARTIAL | Adapter states; cache-independence test |
| Vision reuse | Not specified | Base generation delegates to Transformers | Visual tensors are supplied only on first cached step; profiler records every step | PARTIAL | Adapter diagnostics and `scripts/profile_tcd.py` |
| Batching | Not specified | Not implemented | `batch_size` controls runner grouping, but TCD `generate_batch` is sequential | PARTIAL | `InferenceMethod.generate_batch`; `configs/methods.yaml` |
| Official implementation provenance | Paper links EventHallusion repo | Public tree lacks complete TCD implementation | Local implementation is independently written from equations | UNVERIFIED | Upstream tree and README |
| LLaVA-Video | Paper evaluates Video-LLaVA, not this local adapter | Standard Video-LLaVA Base example only | No local `llava_video` adapter | UNSUPPORTED | `scripts/run_benchmark.py`; compatibility matrix |

## Correctness findings

Before the audit, Qwen's negative frames were downsampled twice: `TCDMethod`
reduced 8 frames to 4, then `_apply_branch_transform` applied another stride to
processor tensors. Besides producing an unintended 2-frame representation, that
mutation could disagree with `image_grid_thw`/`video_grid_thw`. The second
transformation has been removed.

Both adapters construct the two branches independently and retain independent KV
caches. At each timestep they receive the same selected prefix. Transformers'
generation API in the profiled environment did not slice cached text input unless
the caller supplied the next sequence length. A real H200 profile exposed prompt
lengths growing from 23,447 to 23,450 tokens despite a populated cache. The
adapters now explicitly pass only `input_ids[:, -1:]` after prefill while retaining
the complete growing attention mask. New diagnostics record input lengths,
attention-mask lengths, cache presence, and whether visual tensors were supplied
on every step.

The paper does not define behavior when the threshold masks the entire
vocabulary. The local default falls back to the original branch's argmax so a run
does not silently select vocabulary index zero. This is an explicit deviation:
every fallback is counted, and `all_masked_behavior: error` enables strict runs.

## Performance findings

The structural cost is two model forwards per generated token. The pre-audit code
also copied both full-vocabulary logit vectors from GPU to CPU every token and ran
the contrast on NumPy. TCD now keeps both logits and the contrast operation on the
GPU, transferring only the selected scalar token ID. This removes a repeated
device synchronization and large CPU transfer.

The pre-audit LLaVA-OV step decoder also passed `pixel_values` and
`pixel_values_videos` on every cached text step. LLaVA-OneVision runs its vision
tower whenever those tensors are present, so this could recompute visual features
for every generated token in both TCD branches. Both adapters now supply visual
tensors only when their branch cache is empty. The profiler's vision-tower hooks
and `vision_inputs_supplied_steps` counter must confirm one vision pass per branch
on the actual installed Transformers/checkpoint combination.

The first H200 profile after the visual-input fix confirmed one vision pass per
branch, but also found ineffective text-cache slicing: later original-branch
forwards grew from about 4.46 to 6.15 and 8.11 seconds while processing the full
23k-token multimodal prefix. This prompted the explicit one-token cached-input
fix above. A second H200 profile is required before compatibility can be enabled.

A subsequent Qwen H200 profile exposed a model-specific mRoPE contract. Slicing
only cached `input_ids` while forwarding the complete `mm_token_type_ids` made a
one-token hidden state broadcast against 3,171 multimodal positions, failing with
an `o_proj` matrix-shape error. Qwen now slices both sequence-aligned tensors and
supplies image/video grid metadata only during prefill. This path remains disabled
until another real-model profile confirms one-token cached inputs and modality IDs.

Inspection of the installed Transformers 5.16 development API showed that the
generic generation helper slices `input_ids` when `next_sequence_length` is set,
but the adapter's manual loop bypassed Qwen's normal generation-time mRoPE setup.
With a full attention mask and no explicit positions, Qwen reconstructed positions
for the complete 3,171-token prefix and broadcast the one-token hidden state across
that sequence. The adapter now computes 3D prefill positions once, stores
`rope_deltas` independently in each TCD branch, and supplies a `[3, batch, 1]`
position tensor on cached steps. The profiler records position lengths so the H200
run can verify this contract before compatibility is enabled.

The follow-up Qwen2.5-VL-7B H200 profile passed that contract on an eight-frame
real video. The original branch used 9,610 prompt tokens and the four-frame
negative branch used 4,818; every later step used exactly one `input_id`, one
modality ID, and one mRoPE position. Both caches remained populated, and each
vision encoder ran once. Base produced six tokens in 1.57 seconds (3.82 tokens/s,
20.34 GB peak), while TCD produced six tokens in 1.94 seconds (3.09 tokens/s,
20.43 GB peak). Qwen/TCD compatibility is enabled on this evidence; these timing
measurements validate execution and are not benchmark accuracy results.

`scripts/profile_tcd.py` measures model load, video sampling, branch preprocessing,
vision forwards, first/subsequent token forwards, input preparation, prefix sync,
cache update, sequence decode, CUDA synchronization, total time, throughput, and
peak allocated memory. It compares Base and TCD on the same video, prompt, eight
frames, and requested token limits.

No compatible checkpoint, benchmark video, CUDA runtime, or project virtual
environment is present in this local workspace. Consequently, post-fix real-model
timings are not reported here and compatibility remains disabled.

## Final classification

`PARTIALLY CORRECT`

Validation performed locally:

- `python -m compileall -q src scripts tests`: passed.
- TCD/compatibility unit tests: 13 passed.
- `python scripts/smoke_test.py`: passed; deterministic mock only, explicitly not
  a research result.
- Full pytest suite: 4 failures. Two sampler tests could not import optional
  `cv2`; `test_final_results` and `test_videohallucer` expose pre-existing
  non-TCD expectation mismatches. No TCD test failed.
- Real-model profiling: passed remotely on Qwen2.5-VL-7B with an H200 and a real
  VidHalluc video. Cached sequence-aligned inputs were one token, each branch
  supplied vision inputs once, and independent branch caches remained active.

Pipeline logic is covered by local tests and Qwen real-model profiling. Full
benchmark metrics still require complete dataset execution and strict evaluation;
the profiler output itself is not a research accuracy result.
