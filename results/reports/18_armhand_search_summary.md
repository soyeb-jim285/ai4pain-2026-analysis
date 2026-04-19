# 18 — ARM vs HAND ML hyperparameter search

- runtime so far: 529.5s
- configurations attempted: 122
- configurations with full LOSO + val: 102

## Top-15 configurations (by LOSO macro-F1)

| model      | feature_set   | preproc   | params_str                  |   loso_macro_f1_mean |   loso_macro_f1_std |   val_macro_f1 |   val_acc |
|:-----------|:--------------|:----------|:----------------------------|---------------------:|--------------------:|---------------:|----------:|
| lr_en      | top80_cliff   | std       | {"C": 0.1, "l1_ratio": 0.3} |                0.586 |               0.118 |          0.514 |     0.514 |
| lr_en      | top80_cliff   | std       | {"C": 0.1, "l1_ratio": 0.5} |                0.579 |               0.114 |          0.503 |     0.503 |
| lr_l2      | top80_cliff   | std       | {"C": 0.01}                 |                0.579 |               0.111 |          0.497 |     0.497 |
| svm_linear | top80_cliff   | std       | {"C": 0.1}                  |                0.574 |               0.112 |          0.503 |     0.503 |
| lr_l1      | top40_cliff   | std       | {"C": 1.0}                  |                0.572 |               0.099 |          0.462 |     0.462 |
| lr_en      | top80_cliff   | std       | {"C": 0.1, "l1_ratio": 0.7} |                0.572 |               0.111 |          0.507 |     0.507 |
| lr_l1      | top80_cliff   | std       | {"C": 0.1}                  |                0.571 |               0.103 |          0.496 |     0.497 |
| lr_l1      | top40_cliff   | robust    | {"C": 1.0}                  |                0.571 |               0.096 |          0.465 |     0.465 |
| lr_l2      | top80_cliff   | std       | {"C": 0.1}                  |                0.569 |               0.116 |          0.493 |     0.493 |
| lr_en      | top80_cliff   | std       | {"C": 1.0, "l1_ratio": 0.5} |                0.569 |               0.113 |          0.496 |     0.497 |
| lr_en      | top80_cliff   | std       | {"C": 1.0, "l1_ratio": 0.3} |                0.568 |               0.111 |          0.496 |     0.497 |
| lr_l2      | top40_cliff   | std       | {"C": 1.0}                  |                0.567 |               0.101 |          0.469 |     0.469 |
| svm_linear | top80_cliff   | std       | {"C": 1.0}                  |                0.566 |               0.105 |          0.521 |     0.521 |
| lr_l2      | top40_cliff   | robust    | {"C": 1.0}                  |                0.566 |               0.099 |          0.469 |     0.469 |
| lr_l1      | top80_cliff   | std       | {"C": 1.0}                  |                0.566 |               0.110 |          0.510 |     0.510 |

## Ensembles of top-K configs (mean predicted probability)

|      k |   loso_macro_f1 |   loso_acc |   loso_balanced_acc |   val_macro_f1 |   val_acc |   val_balanced_acc |
|-------:|----------------:|-----------:|--------------------:|---------------:|----------:|-------------------:|
|  3.000 |           0.585 |      0.585 |               0.585 |          0.506 |     0.507 |              0.507 |
|  5.000 |           0.568 |      0.568 |               0.568 |          0.510 |     0.510 |              0.510 |
| 10.000 |           0.564 |      0.564 |               0.564 |          0.507 |     0.507 |              0.507 |

## Notes
- Resumable: rerun the script to continue from the last fold.
- Per-fold trail: `results/tables/armhand_search_perfold.csv`.
- Top-K table: `results/tables/armhand_search_top.csv`.