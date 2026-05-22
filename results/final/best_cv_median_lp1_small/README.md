# best_cv_median_lp1_small

Top CV-validated 3-class config under multi-seed (5 seeds × 5 folds) subject CV.
See `results/final/cv_compare/multiseed_top8.csv` and `median_spo2_combo.csv` for the comparison.

## Config
- preproc: `eda_median_lp1` (median win=5 → Butter LP 1 Hz order=3, EDA only)
- feature parquet: `results/tables/all_features_merged_1022_edamdlp1.parquet`
- stage1 features: `bvp_eda_resp_small` (60 ranked BVP+EDA + 10 ranked RESP)
- stage1: logreg L2 C=1.0, subject_z, isotonic, anchor=none
- stage2 features: `bvp_resp_top30`
- stage2: logreg L2 C=1.0, subject_z, isotonic, anchor=none
- decoder: `split_topk`, w0=w1=w2=1.0

## Held-out val (n=432, 12 subj, fixed)
| split | acc | bal_acc | macro_f1 | macro_auc |
|---|---|---|---|---|
| train_loso | 0.608 | 0.608 | 0.608 | 0.793 |
| validation | 0.604 | 0.604 | 0.604 | 0.768 |

## Multi-seed 5×5 CV (n=25 splits)
- acc 0.5750 ± 0.0239 — **best of 10 configs**
- AUC 0.7711 ± 0.0186
- Paired vs `baseline` (edadtlp1 + bvp_eda_core + eda_resp_top30):
  - Δacc = +0.0095 (14/25 wins), t-p = 0.151
  - Not statistically significant at α=0.05 (n=25 underpowered)

## Why not declared winner
Baseline (edadtlp1) has higher held-out val (0.6088) and same CV AUC.
Median preproc wins CV acc trend but no clear AUC gain.
Kept as alternate candidate for ensembling or further validation.
