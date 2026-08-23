# VideoHallucer Dataset Revision Report

Status: `CURRENT-UPSTREAM DATASET; NOT PAPER-V1-COMPATIBLE`

Evidence sources:

- VideoHallucer paper v1: https://arxiv.org/abs/2406.16338v1
- Current upstream: https://github.com/patrick-tssn/VideoHallucer
- Audited upstream commit: `8b785d1680465911cd2ce80c9f652837c0ba2abd`

The bundled annotations are byte-identical to the audited current-upstream
annotations. ORH, SDH, EFH and ENFH each contain 200 valid pairs. TPH contains
176 valid pairs (352 branches). The VideoHallucer paper v1 protocol used 200 TPH pairs. The
upstream README currently reports 376 TPH questions after an October 2025
duplicate-removal change, while its checked-in JSON contains 352 branches.

The SEASON paper was submitted after that stated cleanup and uses VideoHallucer
in Table 1, but it does not publish the annotation commit/hash or pair count.
Therefore it is not currently possible to prove whether SEASON used 176, 188 or
200 TPH pairs from the paper text alone.

No loader defect was found: every current TPH annotation row has one basic and
one hallucination branch. The difference is a dataset revision/provenance issue,
not evidence that 24 pairs should be synthesized.

Two protocols must remain separate:

| Protocol | TPH pairs | Use |
| --- | ---: | --- |
| VideoHallucer-paper-v1 dataset | 200 | Requires the exact archived paper-v1 annotations and videos |
| SEASON-Table-1 dataset | Unverified | Requires author artifact or annotation hash |
| Current-local dataset | 176 | Valid for explicitly labeled local experiments |

The paper-v1 archive is not present in this checkout. Its exact annotation
commit/hash therefore remains unverified.
