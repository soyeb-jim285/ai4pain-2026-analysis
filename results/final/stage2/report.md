# Stage 2 Run
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- normalization: `subject_z`
- anchor mode: `none`
- feature set: `resp_all` (44 features)
- model: `logreg` {'C': 1.0}
- scaler: `robust`
- calibration: `isotonic`

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |   precision_nopain |   recall_nopain |
|:-----------|-----------:|--------------------:|-----------:|-------------------:|----------------:|
| train_loso |   0.522358 |            0.522358 |   0.522358 |           0.522358 |        0.522358 |
| validation |   0.513889 |            0.513889 |   0.513889 |           0.513889 |        0.513889 |