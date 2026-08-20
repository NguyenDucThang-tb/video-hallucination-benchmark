# Upstream provenance

Audit baseline: local `main` at `cad2f352c9da2fe7b1d3b96f54341241fb0d16b3`.
The three upstream repositories were fetched as sparse audit clones and checked
at the exact commits below. Relevant directory comparisons were byte-identical.

| Component | Official URL | Commit | Local path | Relationship | License | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| VideoHallucer evaluators/data annotations | https://github.com/patrick-tssn/VideoHallucer | `8b785d1680465911cd2ce80c9f652837c0ba2abd`, `main`, 2025-12-16 | `external/VideoHallucer/` | Unmodified selected snapshot | MIT | `git diff --no-index` against `evaluations/` returned no differences; local `LICENSE` is MIT. |
| EventHallusion questions/evaluators | https://github.com/Stevetich/EventHallusion | `aa544c21c7cd93b4685423cb94f77ab441f754bc`, `master`, 2025-08-06 | `external/EventHallusion/` | Unmodified selected snapshot | UNVERIFIED: no top-level license at locked commit | Relevant question files and README are byte-identical; redistribution permission must not be inferred. |
| VidHalluc evaluators/DINO-HEAL patch | https://github.com/CyL97/VidHalluc | `e753864f5c2500c38523f97992355e2352bf8732`, `main`, 2025-11-03 | `external/VidHalluc/` | Unmodified selected snapshot | UNVERIFIED: no top-level license at locked commit | `eval/` and `DINO-HEAL/` comparisons returned no differences. |
| Benchmark loaders/evaluators | Sources above | SHAs above | `src/benchmarks/` | Independent adaptation/reimplementation | Repository license; upstream attribution applies to behavior | Implemented against upstream schemas and metric code; not official upstream code. |
| TCD | https://arxiv.org/abs/2409.16597 | paper source inspected | `src/methods/tcd/` | Paper-based reimplementation | Local repository license | No complete token-level TCD source was confirmed in locked EventHallusion tree. |
| DINO-HEAL adapters | https://arxiv.org/abs/2412.03735 | paper plus VidHalluc SHA above | `src/methods/dino_heal/`, `src/models/` | Partial independent adaptation | Local repository license | Current adapters do not reproduce final-layer CLS-to-patch fusion and are disabled. |
| SEASON | https://arxiv.org/abs/2512.04643 | paper source inspected | `src/methods/season/`, `src/models/` | Partial paper-based reimplementation | Local repository license | No official code provenance is claimed; model hooks are incomplete. |

`external/COMMITS.lock` records the same SHAs, branches and URLs. Snapshot code
is retained for audit/attribution only; absence of a license is not permission
to redistribute EventHallusion or VidHalluc code.
