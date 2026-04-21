# Combined Run
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- stage1 norm: `subject_z`
- stage1 feature set: `all` (178 features)
- stage1 model: `xgb` {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 1.0, 'colsample_bytree': 1.0}
- stage1 calibration: `none`
- stage1 anchor: `none` lambda=0.5
- stage2 norm: `subject_robust`
- stage2 feature set: `resp_all` (44 features)
- stage2 model: `logreg` {'C': 1.0}
- stage2 calibration: `none`
- stage2 anchor: `none`
- decoder: `joint_weighted`
- weights: w0=0.8, w1=1.2, w2=1.4

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |
|:-----------|-----------:|--------------------:|-----------:|
| train_loso |   0.564363 |            0.564363 |   0.564363 |
| validation |   0.55787  |            0.55787  |   0.55787  |