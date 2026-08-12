# SEASON Reimplementation

Source: SEASON, CVPR 2026 / arXiv:2512.04643. This is a training-free,
token-level contrastive decoder, not frame dropping.

## Notation and branches

For eight frames V and query Q, one autoregressive context is evaluated under:

- `vO`: original video representation.
- `vS`: spatial negative from Gaussian-corrupted video pixels.
- `vT`: temporal negative produced by layer-wise temporal homogenization.

The same generated prefix is supplied to all three branches at every step.

## Temporal homogenization

At every vision layer l, first compute the ordinary layer output for frame t:

```text
h'[l,t] = E[l](h[l-1,t])
d[l]    = mean_t h'[l,t]
h[l,t]  = (1-beta) h'[l,t] + beta d[l]
```

This is recurrent across all vision layers by default. Applying the equation
only to final embeddings is an approximation and must not be called SEASON in
the main table. Paper grid: `(alpha,beta)` in `{(1.0,0.33),(0.5,0.25)}`.

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

The branch KV caches remain separate. The selected token is appended to all
three branch contexts. Attention for diagnosis belongs to the preceding token,
matching the paper.

## Adapter requirements

A model adapter must expose step logits, decoder attention mapped to exact
frame/patch token ranges, and hooks at every vision layer. If any signal is
unavailable, compatibility is `unsupported`; neither final-feature averaging
nor frame dropping substitutes for SEASON.
