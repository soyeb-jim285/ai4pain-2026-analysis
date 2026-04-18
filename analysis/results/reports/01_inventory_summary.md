# AI4Pain 2026 — Inventory & Integrity Summary

_Generated in 4.9s._

## Shape

- `train`: tensor `(1476, 4, 1000)` (float32), 41 subjects, 1476 segments.
- `validation`: tensor `(432, 4, 1000)` (float32), 12 subjects, 432 segments.

## Subject-level
- No subject overlap between train and validation (good).
- Every subject has exactly 12/12/12 (NoPain/PainArm/PainHand).

## Segment lengths (pre-truncation)
- `train/Bvp`: min=990, median=1022, max=1118, std=15.7; <1000: 36, >1000: 1440, =1000: 0 / 1476
- `train/Eda`: min=990, median=1022, max=1118, std=15.7; <1000: 36, >1000: 1440, =1000: 0 / 1476
- `train/Resp`: min=990, median=1022, max=1118, std=15.7; <1000: 36, >1000: 1440, =1000: 0 / 1476
- `train/SpO2`: min=990, median=1022, max=1118, std=15.7; <1000: 36, >1000: 1440, =1000: 0 / 1476
- `validation/Bvp`: min=1022, median=1022, max=1022, std=0.0; <1000: 0, >1000: 432, =1000: 0 / 432
- `validation/Eda`: min=1022, median=1022, max=1022, std=0.0; <1000: 0, >1000: 432, =1000: 0 / 432
- `validation/Resp`: min=1022, median=1022, max=1022, std=0.0; <1000: 0, >1000: 432, =1000: 0 / 432
- `validation/SpO2`: min=1022, median=1022, max=1022, std=0.0; <1000: 0, >1000: 432, =1000: 0 / 432

- Across all (split,signal) pairs: 144 cases are SHORTER than 1000 (→ tail NaN padding), 7488 are LONGER (→ samples ≥1000 are dropped).

## NaN / Inf
- No interior NaN/inf in any tensor segment (padding NaNs accounted for).
- `train/Bvp`: raw-any-NaN=0, raw-any-inf=0, interior-NaN=0, interior-inf=0, padding-NaN=36 / 1476
- `train/Eda`: raw-any-NaN=0, raw-any-inf=0, interior-NaN=0, interior-inf=0, padding-NaN=36 / 1476
- `train/Resp`: raw-any-NaN=0, raw-any-inf=0, interior-NaN=0, interior-inf=0, padding-NaN=36 / 1476
- `train/SpO2`: raw-any-NaN=0, raw-any-inf=0, interior-NaN=0, interior-inf=0, padding-NaN=36 / 1476

## Constant segments (std < 1e-8)
- 349 constant-segment cases:
  - `train/SpO2`: 254 constant segments
  - `validation/SpO2`: 95 constant segments
  - examples:
    - subj 1 PainArm seg 5 (SpO2): mean=62, std=0
    - subj 4 NoPain seg 3 (SpO2): mean=84, std=0
    - subj 4 NoPain seg 4 (SpO2): mean=84, std=0
    - subj 4 NoPain seg 7 (SpO2): mean=85, std=0
    - subj 4 PainHand seg 2 (SpO2): mean=84, std=0

## Duplicate signal arrays (byte-identical)
- `train/SpO2`: 254 rows in duplicate groups (including originals)
- `validation/SpO2`: 95 rows in duplicate groups (including originals)

## Signal ranges (finite values only)
- `train/Bvp` (arbitrary / ~centered): min=0.458, p1=0.479, median=0.498, p99=0.523, max=0.576
- `train/Eda` (µS, typically 0–30 µS): min=0, p1=0, median=3.74, p99=12.9, max=18.8
- `train/Resp` (arbitrary units): min=-1.49, p1=-0.443, median=-0.0309, p99=0.524, max=1.49
- `train/SpO2` (0–100 %): min=0, p1=0.236, median=94, p99=100, max=100
- `validation/Bvp` (arbitrary / ~centered): min=0.242, p1=0.48, median=0.498, p99=0.522, max=0.605
- `validation/Eda` (µS, typically 0–30 µS): min=0, p1=0, median=4.12, p99=11.6, max=14.6
- `validation/Resp` (arbitrary units): min=-1.49, p1=-0.56, median=-0.0211, p99=0.422, max=1.37
- `validation/SpO2` (0–100 %): min=2, p1=38, median=89, p99=100, max=100

## Outputs
- Tables: `results/tables/inventory_*.csv`
- Plots:  `plots/inventory/*.png`
