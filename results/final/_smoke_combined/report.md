# Combined Run
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- stage1 norm: `subject_robust`
- stage1 feature set: `bvp_eda_core` (98 features)
- stage1 model: `logreg` {'C': 1.0}
- stage1 calibration: `sigmoid`
- stage1 anchor: `none` lambda=0.5
- stage2 norm: `subject_z`
- stage2 feature set: `resp_top20` (20 features)
- stage2 model: `logreg` {'C': 1.0}
- stage2 calibration: `isotonic`
- stage2 anchor: `none`
- decoder: `joint_weighted`
- weights: w0=0.8, w1=1.2, w2=1.4

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |
|:-----------|-----------:|--------------------:|-----------:|
| train_loso |   0.571816 |            0.571816 |   0.571816 |
| validation |   0.532407 |            0.532407 |   0.532407 |