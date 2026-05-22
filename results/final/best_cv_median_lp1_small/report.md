# Combined Run
- feature parquet: `results/tables/all_features_merged_1022_edamdlp1.parquet`
- stage1 norm: `subject_z`
- stage1 feature set: `bvp_eda_resp_small` (70 features)
- stage1 model: `logreg` {'C': 1.0, 'penalty': 'l2', 'l1_ratio': 0.5}
- stage1 calibration: `isotonic`
- stage1 anchor: `none` lambda=0.5
- stage2 norm: `subject_z`
- stage2 feature set: `bvp_resp_top30` (30 features)
- stage2 model: `logreg` {'C': 1.0, 'penalty': 'l2', 'l1_ratio': 0.5}
- stage2 calibration: `isotonic`
- stage2 anchor: `none`
- decoder: `split_topk`
- weights: w0=1.0, w1=1.0, w2=1.0

## Summary

| split      |   accuracy |   balanced_accuracy |   macro_f1 |   macro_auc |
|:-----------|-----------:|--------------------:|-----------:|------------:|
| train_loso |   0.607724 |            0.607724 |   0.607724 |    0.793039 |
| validation |   0.604167 |            0.604167 |   0.604167 |    0.768089 |