# Direct Multiclass Run
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- normalization: `subject_robust`
- feature set: `top80` (80 features)
- model: `xgb` {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 1.0, 'colsample_bytree': 1.0}
- scaler: `std`
- decoder: `exact_counts`

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |
|:-----------|-----------:|--------------------:|-----------:|
| train_loso |   0.530488 |            0.530488 |   0.530488 |
| validation |   0.541667 |            0.541667 |   0.541667 |