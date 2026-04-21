# Combined Run
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- stage1 norm: `subject_z`
- stage1 feature set: `bvp_eda_resp_small` (70 features)
- stage1 model: `logreg` {'C': 1.0}
- stage1 calibration: `isotonic`
- stage1 anchor: `none` lambda=0.5
- stage2 norm: `subject_z`
- stage2 feature set: `resp_all` (44 features)
- stage2 model: `logreg` {'C': 1.0}
- stage2 calibration: `isotonic`
- stage2 anchor: `center`
- decoder: `joint_weighted`
- weights: w0=0.9, w1=1.1, w2=1.2

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |
|:-----------|-----------:|--------------------:|-----------:|
| train_loso |   0.573171 |            0.573171 |   0.573171 |
| validation |   0.569444 |            0.569444 |   0.569444 |