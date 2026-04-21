# Stage 1 Run
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- normalization: `subject_robust`
- feature set: `bvp_eda_core` (98 features)
- model: `xgb` {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 1.0, 'colsample_bytree': 1.0}
- scaler: `std`
- calibration: `sigmoid`
- anchor mode: `center`
- anchor lambda: `0.5`

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |   precision_nopain |   recall_nopain |
|:-----------|-----------:|--------------------:|-----------:|-------------------:|----------------:|
| train_loso |   0.826558 |            0.804878 |   0.804878 |           0.739837 |        0.739837 |
| validation |   0.851852 |            0.833333 |   0.833333 |           0.777778 |        0.777778 |