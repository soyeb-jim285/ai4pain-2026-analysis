# Combined Run
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- stage1 norm: `subject_z`
- stage1 feature set: `all` (189 features)
- stage1 model: `xgb` {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 1.0, 'colsample_bytree': 1.0}
- stage1 calibration: `none`
- stage1 anchor: `none` lambda=0.5
- stage2 norm: `subject_robust`
- stage2 feature set: `resp_top20` (20 features)
- stage2 model: `xgb` {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 1.0, 'colsample_bytree': 1.0}
- stage2 calibration: `isotonic`
- stage2 anchor: `none`
- decoder: `joint_weighted`
- weights: w0=1.0, w1=1.0, w2=1.0

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |
|:-----------|-----------:|--------------------:|-----------:|
| train_loso |   0.581301 |            0.581301 |   0.581301 |
| validation |   0.548611 |            0.548611 |   0.548611 |