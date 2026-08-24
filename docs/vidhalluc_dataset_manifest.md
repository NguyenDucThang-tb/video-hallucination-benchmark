# VidHalluc dataset manifest

Expected source: `chaoyuli/VidHalluc`. Expected annotations are
`ach_binaryqa.json`, `ach_mcq.json`, `tsh.json`, and `sth.json`; expected video
archives are `ACH_videos.zip`, `TSH_videos.zip`, and `STH_videos.zip`.

The dataset is not stored in this checkout. Gadi uses
`/scratch/jp09/dd9648/datasets_video_hallu/vidhalluc/data`.

| Field | Observed on Gadi | Status |
| --- | ---: | --- |
| TSH annotations | 600 | MATCH |
| STH annotations | 445 | MATCH |
| Duplicate annotation IDs | 0 | MATCH |
| Invalid labels | 0 | MATCH |
| Missing or ambiguous video mappings | 44 | PARTIAL |
| Dataset revision | not recorded | UNVERIFIED |

```bash
PYTHONPATH=. ./.venv/bin/python scripts/verify_vidhalluc_dataset.py \
  --dataset-root /scratch/jp09/dd9648/datasets_video_hallu/vidhalluc/data \
  --dataset-source chaoyuli/VidHalluc \
  --dataset-revision '<immutable-HF-revision>' \
  --output results/audit/vidhalluc_dataset_verification.json
```

The verifier records annotation SHA-256 hashes, missing and ambiguous videos,
duplicate IDs, invalid annotations, and usable counts. Until all mappings and
the revision are verified, dataset status remains `PARTIAL`.
