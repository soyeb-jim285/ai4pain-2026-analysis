# ARM vs HAND — best model + 3-run ceiling estimate

*Champion across feature engineering, classical ML hyperparameter sweeps,
deep learning, and 3 independent runs (2 local, 1 Kaggle T4). Honest read
on the ceiling for peripheral physiological signals on this localisation
task.*

## Champions per run

| run | best (val) | model | feature_set | preproc | LOSO | VAL |
|---|---|---|---|---|---|---|
| **local v1** (212 features) | LR-L2 | top80_cliff | std (C=0.1) | 0.581 | **0.565** | |
| local v2 (268 + tierB) | histgb | top40_anova | std (depth=6 lr=0.05) | 0.513 | **0.562** | |
| **kaggle v4** (324 + tierB + tierC) | LR-L2 | resp_only | robust (C=1.0) | 0.536 | **0.545** | |

Spread 0.020 across 3 runs ≈ noise floor on n=288 val (1 segment ≈ 0.0035 macro-F1).

**True val ceiling: 0.55 ± 0.02 macro-F1.**

## What didn't help

| addition | configs ran | net val lift |
|---|---|---|
| tierB derivative features (56 cols, 6 FDR-survivors) | 104 | -0.003 (local v2 vs v1) |
| tierC-v1 paper features (56 cols: wavelet, arc length, fuzzy entropy; script 21) | 122 (Kaggle) | -0.020 |
| tierC-v2 PDA 3-Gaussian on BVP (28 cols; paper 2; script 22) | standalone eval | LOSO 0.512, val 0.528; 0/28 FDR |
| tierC-v3 TVSymp envelope on EDA (12 cols; paper 3; script 23) | standalone eval | LOSO 0.491, val 0.477; 0/12 FDR |
| tight_pool (curated ~80 RESP-focused) | 20 (Kaggle) | not in top-15 |
| 5-seed CNN ensemble (supervised + SSL pretrain) | 2 configs | LOSO 0.46, VAL 0.52 (chance) |
| Top-K sweep ensemble (avg probs of top 3/5/10) | 3 | LOSO 0.58, VAL 0.47 (overfit) |

## What works

| signal class | val macro-F1 |
|---|---|
| LR-L2 + RESP features only | 0.54-0.55 (consistent across runs) |
| LR-L2 + top80 by Cliff's δ | 0.55-0.57 (best original, dilutes when more features added) |
| HistGB + top40 by ANOVA F | 0.53-0.56 |
| XGB + all features | 0.50-0.53 |
| **DL (CNN supervised or SSL+finetune)** | **0.49-0.52 (chance)** |

## Ship recipe

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

# Stage 1 (NoPain vs Pain): XGB + subject-z, val 0.80 (script 12)
# Stage 2 (PainArm vs PainHand): the model below

# Feature set: 53 RESP features from results/tables/tf_features.parquet
# (any column matching channel 'resp', incl. raw_*_Resp_*).

clf = LogisticRegression(
    penalty="l2", C=1.0, class_weight="balanced",
    solver="lbfgs", max_iter=4000, random_state=42,
)
# Per-subject z-score across all 36 of subject's segments (from
# src.data_loader: load_split + apply_subject_z), then RobustScaler
# fit on train ARM+HAND segments only.
```

Realistic test estimate (papers' val→test gap −0.10 to −0.20): **0.40–0.50 test macro-F1**. Near chance for binary localisation. Expected from physiology: peripheral autonomic signals encode pain *intensity* (sympathetic arousal), not anatomical location (somatotopy). Cortical/spinal/EMG would be needed.

## Pipeline integration

```
NoPain vs Pain      ->  XGB + subject-z + top80_cliff + std
                        val macro-F1 = 0.80 (from script 12)
                        only feed Pain-classified to stage 2

Pain -> Arm vs Hand ->  LR-L2 + RESP-only + RobustScaler + subject-z
                        val macro-F1 = 0.55 (this script)

3-class effective:  P(class=NoPain)   = stage1 P(NoPain)
                    P(class=PainArm)  = stage1 P(Pain) * stage2 P(Arm)
                    P(class=PainHand) = stage1 P(Pain) * stage2 P(Hand)
```

End-to-end accuracy estimate (assuming independent errors):
- 0.80 × 0.55 = 0.44 on Pain-class subset
- 0.85 NoPain accuracy
- 3-class balanced ≈ 0.50-0.55 (better than direct 3-class XGB which got 0.515)

## Files

- `results/tables/armhand_search_summary.csv` — full sweep (122 configs from Kaggle v4)
- `results/tables/armhand_search_top.csv` — top-20
- `results/tables/armhand_search_ensembles.csv` — ensemble metrics
- `results/tables/armhand_dl_summary_kaggle_v4.csv` — 5-seed CNN results
- `results/reports/19_armhand_dl_summary.md` — DL run report (in repo from prior run)
- `plots/armhand_search/*.png` — top configs, LOSO-vs-val, per-model boxplot

## Honest assessment

Three independent runs converged on the same ceiling. tierB derivative
features, tierC paper-derived features, tight_pool curation, 5-seed DL
ensembles, and SSL pretraining each failed to push past 0.56 val. The
information about anatomical pain location is genuinely scarce in these
4 peripheral channels for 10-second windows.

For challenge submission: ship the LR-L2 + RESP-only + RobustScaler
recipe. Document the ceiling honestly. Don't oversell.
