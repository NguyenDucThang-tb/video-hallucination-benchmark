# Reproducibility Status

Audit scope: VideoHallucer Base first.

```text
BASELINE NOT REPRODUCED.
DO NOT COMPARE TCD, DINO-HEAL OR SEASON YET.
```

| Component | Status | Reason |
| --- | --- | --- |
| Base | NOT REPRODUCED | Exact SEASON dataset/checkpoint/template/sampling details and local raw artifacts are not verified |
| TCD | NOT VALIDATED | Base gate failed; four-point benchmark grid not executed |
| DINO-HEAL | PARTIAL APPROXIMATION | Current adapters use frame-level evidence rather than verified patch-level alignment |
| SEASON | NOT VALIDATED | LLaVA-OV has local components but no complete artifact-backed end-to-end validation |

Result classes:

| Result type | Available? | Allowed label |
| --- | --- | --- |
| Paper result | Yes, as cited reference only | `PAPER RESULT` |
| Paper-compatible local result | No | none until gate passes |
| Local protocol result | Existing remote runs may qualify after artifact audit | `LOCAL REIMPLEMENTATION RESULTS` |

Required metadata for every future result table: result type, dataset revision,
frame protocol, parser, metric and checkpoint revision. Current mitigation
methods are disabled in `src/models/compatibility.py` to enforce the gate.

Protocol distinction: the original VideoHallucer paper uses baseline-specific
frame/generation defaults, while SEASON Table 1 states a common 8-frame setup for
LLaVA-OV-7B, Qwen2.5-VL-7B and LLaVA-Video-7B. The local eight-frame count is
consistent with SEASON, but exact indices, decoder, resize/crop and checkpoint
revisions remain unverified.
