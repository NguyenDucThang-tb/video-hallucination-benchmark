# SEASON Paper-Grounded Local Reimplementation

Primary reference: **SEASON: Mitigating Temporal Hallucination in Video Large
Language Models via Self-Diagnostic Contrastive Decoding**, CVPR 2026,
arXiv:2512.04643.

There is no public official implementation used by this repository. This code
is a paper-grounded local reimplementation. It is not claimed to be official,
and the LLaVA-OneVision adapter remains disabled in the normal compatibility
matrix until a real-video GPU smoke report passes.

## Algorithm

For the same deterministic eight sampled frames and prompt, SEASON maintains
three independent autoregressive states and KV caches:

- `original`: the unmodified frames.
- `spatial_negative`: deterministic Gaussian-corrupted frames.
- `temporal_homogenized`: the original frames with layer-wise temporal feature
  homogenization in the vision encoder.

Every branch receives the same selected output token at every decoding step.
No branch calls ordinary text generation and no output text is modified after
generation.

### Spatial negative

Paper specification: create a spatially corrupted negative with Gaussian noise
following the visual contrastive decoding setup.

Our interpretation: add zero-mean Gaussian noise in pixel space, clip to the
input range, and seed it from the prompt hash. The default standard deviation
is `0.1` of the pixel range.

Reason: the paper specifies Gaussian corruption but does not publish code or a
checkpoint-specific injection hook.

Potential deviation: the exact noise variance and whether corruption happened
before or after a particular upstream normalization stage are not stated.

### Temporal negative

At every vision layer `l`, the original branch captures the ordinary layer
output and computes its frame mean `d_l`. The temporal branch then applies

```text
h_l,t = (1 - beta) * h'_l,t + beta * d_l
```

The default is `beta=0.33`. The original contexts are stored on CPU after each
layer to reduce GPU memory and copied back only when the matching temporal
layer executes.

Paper specification: `d_l` is precomputed from a standard forward on the
original video and is used at every vision layer.

Our interpretation: LLaVA-OneVision may create multiple equal-sized crops for
each frame. The leading vision batch dimension is grouped as
`[frames, crops_per_frame, ...]`; the mean is taken over the frame dimension
while crop and patch positions are preserved.

Reason: this preserves spatial correspondence across the eight frame groups.

Potential deviation: the paper does not discuss OneVision's dynamic multi-crop
representation. The adapter rejects non-divisible shapes rather than silently
guessing a grouping.

### Self diagnosis

For decoder layers `[20, 21, 22, 23]`, the adapter obtains attention from the
preceding text token to the exact contiguous image-token span for every frame.
It sums heads, selected layers, and visual patches, then applies softmax over
the eight frame scores.

The two Jensen-Shannon divergences are

```text
D_S = JSD(A_original, A_spatial)
D_T = JSD(A_original, A_temporal)
w_S = D_S / (D_S + D_T)
w_T = D_T / (D_S + D_T)
```

When both divergences are numerically zero, both weights are `0.5`. Missing,
mis-shaped, negative, or non-finite attention raises an error.

The first prompt forward is split before its final token. The long multimodal
prefix is cached without decoder attention; only the final prompt token is
then evaluated with `output_attentions=True`. This obtains the paper's
preceding-token query without materializing a full square prompt-attention
matrix.

### Contrastive decoding

At each token, equation 6 is implemented as

```text
z = (1 + alpha) * z_original
    - alpha * (w_S * z_spatial + w_T * z_temporal)
next_token = argmax(z)
```

The default is `alpha=1.0`. Logits remain on the model device, must share the
same one-dimensional vocabulary shape, and must contain only finite values.
Greedy token selection preserves the benchmark policy: no sampling,
temperature zero, and one beam.

## LLaVA-OneVision integration

The target checkpoint is `llava-onevision-qwen2-7b-ov-hf`. The adapter uses
the existing processor and preserves its `input_ids`, attention mask, image
sizes, pixel values, and independent branch caches. Vision inputs are supplied
only on the first prefill for each branch.

Image-token spans are derived from `config.image_token_index`. If the processor
emits one contiguous image-token block per frame, those spans are used directly.
If LLaVA-OneVision emits a single contiguous image-token block for all eight
frames, that block is split in frame order into eight nearly equal spans. A
non-contiguous or otherwise ambiguous layout is a hard failure because
bucketizing unrelated attention positions would no longer implement SEASON.

Decoder attention is switched to the eager implementation before the SEASON
run. This is required because SDPA may not return attention weights.

## Eight-frame protocol

SEASON receives the frames already produced by `src.data.sampler.sample_video`.
It does not read or sample a video itself. `SeasonMethod` rejects any input
whose first dimension is not exactly eight. The existing prediction manifest
therefore remains the source of frame indices for Base, TCD, DINO-HEAL, and
SEASON.

## Configuration

```yaml
season:
  batch_size: 1
  alpha: 1.0
  homogenization_beta: 0.33
  spatial_noise_std: 0.1
  attention_layers: [20, 21, 22, 23]
  expected_frame_count: 8
  epsilon: 1.0e-8
```

The paper also evaluates `(alpha, beta)=(0.5, 0.25)`. Hyperparameter changes
must be recorded in the result metadata and should not be mixed in one table
without an explicit experiment name.

## Validation

Pure numerical and fake-adapter integration tests are included. They are not
model validation. Run the real checkpoint smoke test before enabling SEASON:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/smoke_season.py \
  --video /path/to/one/real/video.mp4 \
  --prompt 'Answer the benchmark question...' \
  --model-path "$MODEL_DIR" \
  --steps 8
```

For a three-sample benchmark diagnostic while compatibility is pending:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_benchmark.py \
  --config /tmp/experiment_llava_ov_season_videohallucer.yaml \
  --limit 3 --allow-unvalidated --debug-errors
```

Do not run the full benchmark until all five VideoHallucer tasks have no
generation failures and the smoke JSON contains non-empty Base and SEASON
outputs.

## Known limitations

- The official authors have not released code used by this repository.
- Gaussian noise placement and variance are reconstructed from the paper.
- Equal OneVision crop/token counts per sampled frame are adapter assumptions
  when the processor emits one merged visual-token block; this strategy is
  recorded in prediction diagnostics as `visual_token_span_strategy`.
- CPU storage for original per-layer contexts favors memory safety over speed.
- Qwen2.5-VL's older SEASON path remains partial and disabled; this work targets
  LLaVA-OneVision.
- A GPU smoke artifact is still required before normal compatibility is marked
  runnable.

The unrelated positive-feature experiment in `positive_features.py` is not
part of SEASON and is not invoked by `SeasonMethod`.
