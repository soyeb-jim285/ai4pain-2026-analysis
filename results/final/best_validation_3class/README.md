# best_validation_3class

Frozen copy of top-val 3-class run, preserved for reference.

**Source**: `results/final/preproc_featset_sweep/eda_detrend_lp1__bvp_eda_core__eda_resp_top30/`

## Config
- preproc: `eda_detrend_lp1` (EDA detrend + 1 Hz LP)
- feature parquet: `results/tables/all_features_merged_1022_edadtlp1.parquet`
- stage1: `bvp_eda_core` (98 feats), subject_z, logreg L2 C=1.0, isotonic, anchor=none
- stage2: `eda_resp_top30` (30 feats), subject_z, logreg L2 C=1.0, isotonic, anchor=none
- decoder: `split_topk`, w0=w1=w2=1.0

## Val metrics (n=432, 12 subj, balanced)
| metric | value |
|---|---|
| accuracy | 0.6088 |
| balanced acc | 0.6088 |
| macro F1 | 0.6088 |
| stage1 AUC (pain vs nopain) | 0.9033 |
| stage2 AUC (arm vs hand) | 0.5815 |
| macro OvR AUC (3-class) | 0.7798 |

LOSO train acc 0.5786 (n=1476, 41 subj). Val > LOSO by +3pp, no overfit.

Bottleneck: stage2 arm vs hand AUC 0.58.
