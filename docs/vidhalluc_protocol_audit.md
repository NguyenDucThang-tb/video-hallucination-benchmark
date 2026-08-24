# VidHalluc protocol audit

Audit baseline: repository commit `bfde71a927e954e6b29be0aa2686bd8ab3271cd8`.
Upstream VidHalluc snapshot: `e753864f5c2500c38523f97992355e2352bf8732`.

The local pipeline is a paper-based reimplementation; official upstream
implementation was not copied directly. Selected upstream files are retained under
`external/VidHalluc` for parity tests and provenance.

| Item | VidHalluc paper | SEASON paper | Official code | Local code | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset revision | Public VidHalluc release | Uses VidHalluc | Annotations distributed separately | Correct 600/445 counts observed, but no immutable revision recorded | UNVERIFIED | Dataset verifier and Gadi audit |
| TSH count | 600 | Uses benchmark score | `tsh.json` | Gadi inventory reports 600 | MATCH | Audit summary |
| STH count | 445 | Uses benchmark score | `sth.json` | Gadi inventory reports 445 | MATCH | Audit summary |
| Video ID mapping | Released concatenated videos | Not redefined | Exact `{video}.mp4` search | Exact filename/stem; 44 unresolved/ambiguous mappings remain | PARTIAL | Loader and verifier |
| TSH prompt | Action-order task | Uses benchmark protocol | Appends sorting instruction | Same string and action order | MATCH | Protocol test |
| STH prompt | Scene-change plus locations | Uses benchmark protocol | Fixed instruction | Same string | MATCH | Protocol test |
| Frame count | Public inference caps at 32 after about 1 FPS | Controlled 8-frame comparisons | `for_get_frames_num=32` | TSH/STH use exactly 8 frames for the SEASON comparison protocol | MISMATCH | Config and protocol test |
| Frame sampling | Chronological, about 1 FPS, uniform cap | Common frames per comparison | Decord/OpenCV hybrid | Eight chronological rounded-linspace frames, indices recorded | MISMATCH | Sampler and manifests |
| Chat template | Backbone-specific LLaVA stack | Same settings per group | `modalities="video"` | Hugging Face LLaVA-OV/Qwen templates | MISMATCH | Adapter diagnostics |
| Temperature | 0 | 0 | Greedy, one beam, `top_p=0.1` | Greedy, one beam, max 128 tokens | PARTIAL | Resolved config |
| Sampling | Deterministic | Deterministic | `do_sample=False` | `do_sample=False` | MATCH | `GenerationConfig` |
| TSH parser | Accuracy over all annotations | Uses benchmark result | Accepts A/B/AB/BA and action-order text | Compatibility parser preserves Python `None` | MATCH | Vendored parity test |
| `None` handling | Non-match stays in denominator | Not redefined | String `None` does not match AB/BA | Python `None`, never Boolean label | MATCH | Metric tests |
| TSH denominator | All 600 entries | Uses benchmark result | Correct/total | All supplied records | MATCH | Evaluator tests and Gadi audit |
| STH MCC | Required | Uses benchmark result | MCC then `((MCC+1)/2)^2` | Same | MATCH | Evaluator code |
| STH SimCSE | Description component | Uses benchmark result | `princeton-nlp/sup-simcse-roberta-large` | Requires explicit checkpoint | NOT_EXECUTED | No checkpoint-backed artifact committed |
| STH weighting | 0.6 classification + 0.4 description | Uses benchmark result | Same | Same | MATCH | Evaluator tests |
| Base status | Matching backbone/protocol required | Base comparison gate | Original LLaVA stack | Different HF adapters and incomplete dataset provenance/mapping | PARTIAL | Items above |

```text
TSH_STATUS: PARTIAL
STH_STATUS: NOT_EXECUTED (official SimCSE component)
DATASET_STATUS: PARTIAL
PROMPT_STATUS: MATCH
INFERENCE_SETUP_STATUS: PARTIAL
PARSER_STATUS: MATCH
METRIC_STATUS: PARTIAL
BASE_STATUS: BASE_REPRODUCTION_PARTIAL
REPRODUCTION_STATUS: PARTIALLY PAPER-FAITHFUL
```

Do not compare TCD, DINO-HEAL, or SEASON with paper numbers as reproduction
results until dataset provenance, video mappings, backbone equivalence and the
official STH description component are validated.
