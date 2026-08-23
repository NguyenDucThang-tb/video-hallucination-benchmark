# TCD Grid Search Report

Status: `NOT EXECUTED - BLOCKED BY BASELINE GATE`

The SEASON appendix reports the following TCD comparison grid per benchmark:

| Negative frame count | Alpha | Beta |
| ---: | ---: | ---: |
| 2 | 1.0 | 0.1 |
| 2 | 0.5 | 0.5 |
| 4 | 1.0 | 0.1 |
| 4 | 0.5 | 0.5 |

The TCD paper describes chronological downsampling from an original sampled
video to a contrastive video with fewer sampled frames. Its ablation names 1,
4, 8 and 16 frames, so local configuration uses the unambiguous name
`negative_frame_count`; it is a target count, not a stride.

No configuration has been selected. Selection on a benchmark test set would be
test-set tuning and must be labeled as such. All future runs must be retained in
`results/tcd_grid/tcd_all_configs.csv`; the selected file stays empty until the
selection rule, dataset revision and Base gate are resolved.
