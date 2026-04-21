# Modality Tables — Final Pipeline (3-class) — Grid-Optimized per Modality
- feature parquet: `results/tables/all_features_merged_1022.parquet`
- optimization: `True`, selection: train_loso_macro_f1

## Table 1 — Validation Overall Metrics by Modality

| split      | modality   |   n_features |   accuracy |   balanced_accuracy |   macro_f1 |
|:-----------|:-----------|-------------:|-----------:|--------------------:|-----------:|
| validation | all        |          212 |   0.548611 |            0.548611 |   0.548611 |
| validation | bvp        |           62 |   0.511574 |            0.511574 |   0.511574 |
| validation | eda        |           55 |   0.511574 |            0.511574 |   0.511574 |
| validation | bvp_eda    |          117 |   0.506944 |            0.506944 |   0.506944 |
| validation | resp       |           53 |   0.409722 |            0.409722 |   0.409722 |
| validation | spo2       |           42 |   0.395833 |            0.395833 |   0.395833 |

## Table 2 — Validation Classwise Metrics by Modality

| split      | modality   | class    |   precision |   recall |       f1 |   support |
|:-----------|:-----------|:---------|------------:|---------:|---------:|----------:|
| validation | bvp        | NoPain   |    0.673611 | 0.673611 | 0.673611 |       144 |
| validation | bvp        | PainArm  |    0.4375   | 0.4375   | 0.4375   |       144 |
| validation | bvp        | PainHand |    0.423611 | 0.423611 | 0.423611 |       144 |
| validation | eda        | NoPain   |    0.645833 | 0.645833 | 0.645833 |       144 |
| validation | eda        | PainArm  |    0.388889 | 0.388889 | 0.388889 |       144 |
| validation | eda        | PainHand |    0.5      | 0.5      | 0.5      |       144 |
| validation | resp       | NoPain   |    0.423611 | 0.423611 | 0.423611 |       144 |
| validation | resp       | PainArm  |    0.388889 | 0.388889 | 0.388889 |       144 |
| validation | resp       | PainHand |    0.416667 | 0.416667 | 0.416667 |       144 |
| validation | spo2       | NoPain   |    0.486111 | 0.486111 | 0.486111 |       144 |
| validation | spo2       | PainArm  |    0.333333 | 0.333333 | 0.333333 |       144 |
| validation | spo2       | PainHand |    0.368056 | 0.368056 | 0.368056 |       144 |
| validation | bvp_eda    | NoPain   |    0.694444 | 0.694444 | 0.694444 |       144 |
| validation | bvp_eda    | PainArm  |    0.395833 | 0.395833 | 0.395833 |       144 |
| validation | bvp_eda    | PainHand |    0.430556 | 0.430556 | 0.430556 |       144 |
| validation | all        | NoPain   |    0.743056 | 0.743056 | 0.743056 |       144 |
| validation | all        | PainArm  |    0.465278 | 0.465278 | 0.465278 |       144 |
| validation | all        | PainHand |    0.4375   | 0.4375   | 0.4375   |       144 |

## Table 3 — Train-LOSO Overall Metrics by Modality

| split      | modality   |   n_features |   accuracy |   balanced_accuracy |   macro_f1 |
|:-----------|:-----------|-------------:|-----------:|--------------------:|-----------:|
| train_loso | all        |          212 |   0.592141 |            0.592141 |   0.592141 |
| train_loso | bvp_eda    |          117 |   0.584688 |            0.584688 |   0.584688 |
| train_loso | bvp        |           62 |   0.562331 |            0.562331 |   0.562331 |
| train_loso | eda        |           55 |   0.507453 |            0.507453 |   0.507453 |
| train_loso | spo2       |           42 |   0.436992 |            0.436992 |   0.436992 |
| train_loso | resp       |           53 |   0.411247 |            0.411247 |   0.411247 |

## Table 4 — Selected Config per Modality

| modality   |   n_features | stage1_model   | stage1_norm    | stage1_scaler   | stage1_calibration   | stage1_anchor_mode   |   stage1_anchor_lambda |   stage1_logreg_c | stage2_model   | stage2_norm    | stage2_scaler   | stage2_calibration   | stage2_anchor_mode   |   stage2_logreg_c |   w0 |   w1 |   w2 |   selection_train_loso_macro_f1 |   val_macro_f1 |
|:-----------|-------------:|:---------------|:---------------|:----------------|:---------------------|:---------------------|-----------------------:|------------------:|:---------------|:---------------|:----------------|:---------------------|:---------------------|------------------:|-----:|-----:|-----:|--------------------------------:|---------------:|
| bvp        |           62 | logreg         | subject_robust | std             | sigmoid              | none                 |                    0.5 |                 1 | xgb            | subject_z      | robust          | isotonic             | none                 |                 1 |  0.6 |  1.4 |  1.6 |                        0.562331 |       0.511574 |
| eda        |           55 | logreg         | subject_z      | std             | sigmoid              | none                 |                    0.5 |                 1 | logreg         | subject_robust | robust          | isotonic             | none                 |                 1 |  0.8 |  1.2 |  1.4 |                        0.507453 |       0.511574 |
| resp       |           53 | xgb            | subject_robust | std             | sigmoid              | center               |                    0.5 |                 1 | logreg         | subject_robust | robust          | isotonic             | none                 |                 1 |  0.8 |  1.2 |  1.4 |                        0.411247 |       0.409722 |
| spo2       |           42 | xgb            | subject_robust | std             | sigmoid              | center               |                    0.5 |                 1 | logreg         | subject_z      | robust          | isotonic             | none                 |                 1 |  0.8 |  1.2 |  1.4 |                        0.436992 |       0.395833 |
| bvp_eda    |          117 | logreg         | subject_z      | std             | sigmoid              | none                 |                    0.5 |                 1 | logreg         | subject_robust | robust          | isotonic             | none                 |                 1 |  1   |  1   |  1   |                        0.584688 |       0.506944 |
| all        |          212 | xgb            | subject_z      | std             | sigmoid              | none                 |                    0.5 |                 1 | logreg         | subject_z      | robust          | isotonic             | none                 |                 1 |  0.6 |  1.4 |  1.6 |                        0.592141 |       0.548611 |