# SEASON Validation Report

Status: `NOT VALIDATED`

Primary source: https://arxiv.org/abs/2512.04643

The local LLaVA-OV path implements three independent branches, Gaussian spatial
negatives, layer-wise temporal homogenization hooks, selected decoder attention
layers `[20, 21, 22, 23]`, JSD diagnostic weights and token-level contrastive
logits. Unit tests cover the mathematical helpers.

End-to-end validation is still missing for:

- exact visual-token-to-frame mapping for the target checkpoint;
- preceding-token attention at every cached decoding step;
- homogenization after every actual vision encoder layer;
- three independent and persistent KV caches;
- non-degenerate per-token `wS` and `wT` traces;
- EOS behavior and real-video output on the target GPU environment;
- the full `(alpha, beta)` grid per benchmark.

The Qwen path remains partial, and LLaVA-Video has no validated adapter. SEASON
must not be called `Ours`, paper-faithful, or included in a paper-compatible
results table until these checks and the Base gate pass.
