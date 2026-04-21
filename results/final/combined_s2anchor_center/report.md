# Combined Run
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- stage1 norm: `subject_robust`
- stage1 feature set: `bvp_eda_core` (98 features)
- stage1 model: `xgb` {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 1.0, 'colsample_bytree': 1.0}
- stage1 calibration: `sigmoid`
- stage1 anchor: `center` lambda=0.5
- stage2 norm: `subject_z`
- stage2 feature set: `resp_all` (44 features)
- stage2 model: `logreg` {'C': 1.0}
- stage2 calibration: `isotonic`
- stage2 anchor: `center`
- decoder: `joint_weighted`
- weights: w0=0.8, w1=1.2, w2=1.4

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |
|:-----------|-----------:|--------------------:|-----------:|
| train_loso |   0.559621 |            0.559621 |   0.559621 |
| validation |   0.587963 |            0.587963 |   0.587963 |