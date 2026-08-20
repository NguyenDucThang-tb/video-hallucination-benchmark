# SEASON Partial Reimplementation

Source: SEASON, CVPR 2026 / arXiv:2512.04643. This is a training-free,
token-level contrastive decoder, not frame dropping.

## Notation and branches

For eight frames V and query Q, the paper evaluates:

- `vO`: original video representation.
- `vS`: spatial negative from Gaussian-corrupted video pixels.
- `vT`: temporal negative produced by layer-wise temporal homogenization.

The same generated prefix is supplied to all three branches at every step.

## Separate positive-feature extension

The SEASON paper does not define a positive branch. The local experimental
module injects foreground persistence and directed temporal
evidence into post-encoder/projector visual embeddings. For visual features
`V[t,p]`, foreground saliency `F[t,p]`, and persistence
`P[p] = mean_t F[t,p]`:

```text
S[t,p]  = alpha_pos * F[t,p] + alpha_spatial * P[p]
E[t,p]  = normalize(V[t,p] - V[t-1,p]) * ||V[t,p]||
V'[t,p] = V[t,p] * (1 + S[t,p]) + beta_temporal * E[t,p]
```

`E[0,p]` is zero. The evidence term is multiplied by the foreground mask before
normalization, so background patches are excluded from motion evidence. The
implementation is `src.methods.season.enhance_visual_features`. It is not
called by `SeasonMethod` and must not be used as evidence of SEASON fidelity.

## Temporal homogenization

At every vision layer l, first compute the ordinary layer output for frame t:

```text
h'[l,t] = E[l](h[l-1,t])
d[l]    = mean_t h'[l,t]
h[l,t]  = (1-beta) h'[l,t] + beta d[l]
```

This must recur across vision layers. The current Qwen adapter instead modifies
pixel input once, so it is partial and disabled. Paper grid:
`(alpha,beta)` in `{(1.0,0.33),(0.5,0.25)}`.

## Token-level self diagnosis

For decoder attention layers J = `[20,21,22,23]`, sum heads, selected layers,
and visual patches in each frame for the preceding text token, then softmax
over frames:

```text
A_frame(v) = softmax_t sum_k (sum_{j in J} A_j)(y[i-1], v[t,k])
D_S = JSD(A_frame(vO), A_frame(vS))
D_T = JSD(A_frame(vO), A_frame(vT))
w_S = D_S / (D_S + D_T)
w_T = D_T / (D_S + D_T)
```

When both divergences are numerically zero, the implementation uses equal
weights and logs the degenerate diagnosis.

## Contrastive decoding

At each output token:

```text
z = (1 + alpha) zO - alpha * (w_S zS + w_T zT)
y_i = argmax(z)
```

The selected token must be appended to all three branch contexts and KV caches
must remain separate. The current adapters have not demonstrated exact visual
token ranges and preceding-token attention on a real checkpoint.

## Adapter requirements

A model adapter must expose step logits, decoder attention mapped to exact
frame/patch token ranges, and hooks at every vision layer. If any signal is
unavailable, compatibility is `unsupported`; neither final-feature averaging
nor frame dropping substitutes for SEASON.

Video frames are sampled through the OpenCV-backed
`src.data.sampler.sample_video` path. This is slower than some specialized
video readers but avoids decoder-specific frame loading failures in the
benchmark protocol.
