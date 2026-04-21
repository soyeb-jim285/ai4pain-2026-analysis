# Stage 2 Normalization Sweep

- features: `results/tables/all_features_merged_1022.parquet`
- feature set: `resp_top20`
- models: ['logreg']
- scaler: `robust`  calibration: `isotonic`

## Results (sorted by LOSO macro-F1)

| strategy       | model   |   n_features |   loso_macro_f1 |   loso_acc |   loso_per_subject_std |   loso_per_subject_min |   val_macro_f1 |   val_acc |   val_per_subject_std |   val_per_subject_min |
|:---------------|:--------|-------------:|----------------:|-----------:|-----------------------:|-----------------------:|---------------:|----------:|----------------------:|----------------------:|
| subject_robust | logreg  |           20 |          0.5589 |     0.5589 |                 0.0995 |                 0.3333 |         0.5208 |    0.5208 |                0.0908 |                0.3333 |
| nopain_z       | logreg  |           20 |          0.5325 |     0.5325 |                 0.0936 |                 0.3333 |         0.5000 |    0.5000 |                0.0833 |                0.3333 |