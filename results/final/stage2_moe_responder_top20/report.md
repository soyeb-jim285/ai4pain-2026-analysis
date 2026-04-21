# Stage 2 MoE-by-Responder-Cluster Sweep

- model: `logreg`  feature_set: `resp_top20`
- scaler: `robust`  calibration: `isotonic`
- norm: `subject_z` (fixed)

## Results (sorted by LOSO macro-F1)

|   k |   loso_macro_f1 |   loso_acc |   loso_per_subject_std |   loso_per_subject_min |   val_macro_f1 |   val_acc |   val_per_subject_std |   val_per_subject_min | stratum_counts        |
|----:|----------------:|-----------:|-----------------------:|-----------------------:|---------------:|----------:|----------------------:|----------------------:|:----------------------|
|   1 |          0.5569 |     0.5569 |                 0.0982 |                 0.4167 |         0.5486 |    0.5486 |                0.0533 |                0.5000 | {0: 41}               |
|   2 |          0.5427 |     0.5427 |                 0.1166 |                 0.3333 |         0.5486 |    0.5486 |                0.1046 |                0.3333 | {0: 20, 1: 21}        |
|   3 |          0.5407 |     0.5407 |                 0.1151 |                 0.3333 |         0.5139 |    0.5139 |                0.0748 |                0.4167 | {0: 14, 1: 13, 2: 14} |