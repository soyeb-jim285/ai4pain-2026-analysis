# 18 — ARM vs HAND ML hyperparameter search

- runtime so far: 923.4s
- configurations attempted: 102
- configurations with full LOSO + val: 102

## Top-15 configurations (by LOSO macro-F1)

| model      | feature_set   | preproc   | params_str                  |   loso_macro_f1_mean |   loso_macro_f1_std |   val_macro_f1 |   val_acc |
|:-----------|:--------------|:----------|:----------------------------|---------------------:|--------------------:|---------------:|----------:|
| svm_linear | top80_cliff   | std       | {"C": 0.1}                  |                0.597 |               0.108 |          0.524 |     0.524 |
| lr_l2      | top80_cliff   | std       | {"C": 0.01}                 |                0.586 |               0.106 |          0.517 |     0.517 |
| lr_l2      | top40_anova   | std       | {"C": 1.0}                  |                0.581 |               0.111 |          0.531 |     0.531 |
| lr_l2      | top80_cliff   | std       | {"C": 0.1}                  |                0.581 |               0.107 |          0.565 |     0.566 |
| lr_l2      | top40_anova   | robust    | {"C": 1.0}                  |                0.578 |               0.108 |          0.524 |     0.524 |
| svm_linear | top80_cliff   | std       | {"C": 1.0}                  |                0.577 |               0.109 |          0.499 |     0.500 |
| lr_en      | top80_cliff   | std       | {"C": 0.1, "l1_ratio": 0.3} |                0.576 |               0.108 |          0.548 |     0.549 |
| lr_l1      | top80_cliff   | robust    | {"C": 1.0}                  |                0.575 |               0.110 |          0.562 |     0.562 |
| lr_en      | top80_cliff   | std       | {"C": 1.0, "l1_ratio": 0.5} |                0.574 |               0.105 |          0.551 |     0.552 |
| lr_l1      | top80_cliff   | std       | {"C": 1.0}                  |                0.574 |               0.107 |          0.558 |     0.559 |
| lr_en      | top80_cliff   | std       | {"C": 0.1, "l1_ratio": 0.5} |                0.573 |               0.110 |          0.544 |     0.545 |
| lr_en      | top80_cliff   | std       | {"C": 1.0, "l1_ratio": 0.3} |                0.572 |               0.105 |          0.555 |     0.556 |
| lr_l1      | top40_anova   | std       | {"C": 1.0}                  |                0.572 |               0.110 |          0.528 |     0.528 |
| lr_en      | top80_cliff   | std       | {"C": 1.0, "l1_ratio": 0.7} |                0.572 |               0.107 |          0.558 |     0.559 |
| lr_l2      | top80_cliff   | robust    | {"C": 1.0}                  |                0.571 |               0.107 |          0.551 |     0.552 |

## Ensembles of top-K configs (mean predicted probability)

|      k |   loso_macro_f1 |   loso_acc |   loso_balanced_acc |   val_macro_f1 |   val_acc |   val_balanced_acc |
|-------:|----------------:|-----------:|--------------------:|---------------:|----------:|-------------------:|
|  3.000 |           0.602 |      0.602 |               0.602 |          0.524 |     0.524 |              0.524 |
|  5.000 |           0.601 |      0.601 |               0.601 |          0.520 |     0.521 |              0.521 |
| 10.000 |           0.586 |      0.586 |               0.586 |          0.524 |     0.524 |              0.524 |

## Notes
- Resumable: rerun the script to continue from the last fold.
- Per-fold trail: `results/tables/armhand_search_perfold.csv`.
- Top-K table: `results/tables/armhand_search_top.csv`.