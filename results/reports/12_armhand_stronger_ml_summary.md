# Arm-vs-Hand — stronger ML (script 12)

- Last saved after: **all models complete**
- n_features (merged): **1231**
- n_arm_hand_segments: train=984, val=288

## LOSO macro-F1 (chance = 0.50)

| model | macro-F1 (mean ± std) | balanced acc | n_features |
|---|---|---|---|
| logreg | 0.536 ± 0.111 | 0.538 | 1231 |
| rf | 0.522 ± 0.102 | 0.526 | 1231 |
| xgb | 0.517 ± 0.097 | 0.521 | 1231 |
| subwindow_logreg | 0.526 ± 0.091 | 0.537 | 252 |

## Validation split (chance = 0.50)

| model | val macro-F1 | val acc | val balanced acc |
|---|---|---|---|
| logreg | 0.524 | 0.524 | 0.524 |
| rf | 0.502 | 0.503 | 0.503 |
| xgb | 0.469 | 0.469 | 0.469 |
| subwindow_logreg | 0.537 | 0.538 | 0.538 |

## Notes
- All feature-based models use per-subject z-scored merged features.
- The CNN uses per-subject z-scored raw (4, 1000) tensor.
- Every fold uses strict LOSO on the held-out subject.
- Runtime so far: 493.8s