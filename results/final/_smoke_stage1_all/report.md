# Stage 1 Run
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- normalization: `subject_z`
- feature set: `all` (178 features)
- model: `xgb` {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 1.0, 'colsample_bytree': 1.0}
- scaler: `std`
- calibration: `none`
- anchor mode: `none`
- anchor lambda: `0.5`

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |   precision_nopain |   recall_nopain |
|:-----------|-----------:|--------------------:|-----------:|-------------------:|----------------:|
| train_loso |   0.827913 |            0.806402 |   0.806402 |           0.74187  |        0.74187  |
| validation |   0.828704 |            0.807292 |   0.807292 |           0.743056 |        0.743056 |