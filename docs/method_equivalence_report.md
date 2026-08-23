# Method equivalence report

| Method | Official evidence | Local relationship | Confirmed aligned behavior | Material gap | Research-table status |
| --- | --- | --- | --- | --- | --- |
| Base | Model APIs only | Local adapters | Eight pre-sampled frames, fixed prompt, greedy generation | Current TPH revision and universal 8-frame protocol do not reproduce paper-v1 VideoHallucer | Local protocol only |
| TCD | arXiv:2409.16597 | Reimplementation from paper; complete source not confirmed in EventHallusion | Chronological subset of the same eight frames; synchronized generated prefix; `(1+a)z_ori-a*z_con`; `b*max(z_ori)` mask; original-logit fallback if all masked | Base gate failed and the four-point comparison grid has not run | Not validated / blocked |
| DINO-HEAL | VidHalluc `e753864`; arXiv:2412.03735 | Partial adaptation | DINO load is mandatory and failure raises; configured fusion weights are 0.3/0.7; no CLIP-only fallback is accepted | Paper uses last-layer, head-averaged CLS-to-patch attention aligned to visual patches. Local adapters derive token norms, reduce to frame scalars and scale broad feature spans | Partial, not paper-faithful, unsupported / N/A |
| SEASON | arXiv:2512.04643 | Partial paper-based reimplementation | LLaVA-OV has three token-level branches, layer hooks and contrastive-logit/JSD helpers | Exact frame-token mapping, attention/caches and real-video traces remain unverified; Qwen remains partial | Not validated / blocked |
| Positive feature enhancement | User-proposed extension | Independent experimental module | Foreground persistence and directed motion helpers are unit tested | It is not a SEASON paper component and is not invoked by `SeasonMethod` | Excluded from SEASON claims/table |

## TCD details

The local negative branch is formed only from positions in the common sampled
eight-frame array; it does not reopen or resample the source video. Original
and negative branches receive the same generated token prefix at every decode
step. These algorithmic properties have unit tests. They do not establish
checkpoint-level correctness, KV-cache compatibility or benchmark validity.

Required provenance statement: **TCD source implementation not confirmed in
EventHallusion; local code is a reimplementation based on paper.**

## DINO-HEAL details

The upstream/paper pipeline obtains final-layer DINO attention, averages heads,
selects CLS-to-patch attention, reshapes/interpolates it to the target visual
patch grid and fuses it at patch level. Current `generate_dino_heal` paths in
both model adapters pool DINO patch token norms into one value per frame. This
cannot be described as the official implementation. The code is retained for
further engineering but compatibility is disabled.

## SEASON details

SEASON uses original video representation, Gaussian spatial negative and
layer-wise temporal-homogenized negative. The paper does not specify the local
positive-feature module. Paper-faithful support requires exact vision-layer
hooks, visual-token ranges in decoder attention, independent branch caches and
real-model validation. Those contracts are not currently demonstrated, so
`supports_*` properties inside an adapter are insufficient to make a job ready.
