# Tier-A3 — Stage-1 calibration & ARM/HAND drift

Tests whether the stage-1 (pain vs no-pain) classifier's pain probability differs systematically between PainArm and PainHand segments, in which case it is a free meta-feature for stage-2.

## 1. Per-subject ARM vs HAND drift (paired Wilcoxon)

| split | n_subj | mean(ARM) | mean(HAND) | Δ (ARM-HAND) | Wilcoxon W | p | t-test p | Cliff's δ | sign consistency (ARM>HAND) |
|---|---|---|---|---|---|---|---|---|---|
| train | 41 | 0.875 | 0.877 | -0.0012 | 415.0 | 0.848 | 0.928 | -0.026 | 0.46 |
| validation | 12 | 0.882 | 0.872 | +0.0097 | 21.0 | 0.176 | 0.525 | +0.005 | 0.75 |

## 2. Correlation of pain_prob with ARM(0)/HAND(1) (pain segments only)

| split | n | Pearson r | p | point-biserial r | p |
|---|---|---|---|---|---|
| train | 984 | +0.0026 | 0.935 | +0.0026 | 0.935 |
| validation | 288 | -0.0208 | 0.725 | -0.0208 | 0.725 |

- Per-subject (train): median |point-biserial r| = 0.171; 1/41 subjects reach p<0.05 individually.

## 3. pain_prob alone as ARM vs HAND classifier (LOSO)

- macro-F1 = **0.388 ± 0.080** (chance = 0.50)
- balanced acc = **0.461 ± 0.067**
- n_folds = 41

## 4. Recommendation

**Skip pain_prob as a meta-feature.** The stage-1 probability does not differ systematically between ARM and HAND (p=0.848, Cliff's δ=-0.026); stand-alone macro-F1 = 0.388 is at or below chance.

## Files

- `results/tables/tierA3_stage1_probabilities.csv`
- `results/tables/tierA3_stage1_armhand_drift.csv`
- `results/tables/tierA3_stage1_correlation.csv`
- `results/tables/tierA3_stage1_as_feature_loso.csv`
- `plots/tierA3_stage1/*.png`