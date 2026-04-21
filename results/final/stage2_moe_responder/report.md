# Stage 2 MoE-by-Responder-Cluster Sweep

- model: `logreg`  feature_set: `resp_all`
- scaler: `robust`  calibration: `isotonic`
- norm: `subject_z` (fixed)

## Results (sorted by LOSO macro-F1)

|   k |   loso_macro_f1 |   loso_acc |   loso_per_subject_std |   loso_per_subject_min |   val_macro_f1 |   val_acc |   val_per_subject_std |   val_per_subject_min | stratum_counts               |
|----:|----------------:|-----------:|-----------------------:|-----------------------:|---------------:|----------:|----------------------:|----------------------:|:-----------------------------|
|   2 |          0.5305 |     0.5305 |                 0.1069 |                 0.3333 |         0.5625 |    0.5625 |                0.0601 |                0.5000 | {0: 20, 1: 21}               |
|   1 |          0.5224 |     0.5224 |                 0.0974 |                 0.3333 |         0.5139 |    0.5139 |                0.0889 |                0.3333 | {0: 41}                      |
|   3 |          0.5203 |     0.5203 |                 0.1190 |                 0.2500 |         0.5347 |    0.5347 |                0.1046 |                0.2500 | {0: 14, 1: 13, 2: 14}        |
|   4 |          0.5061 |     0.5061 |                 0.1031 |                 0.3333 |         0.5278 |    0.5278 |                0.0393 |                0.5000 | {0: 10, 1: 10, 2: 10, 3: 11} |