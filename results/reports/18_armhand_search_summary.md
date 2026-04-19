# 18 — ARM vs HAND ML hyperparameter search

- runtime so far: 3797.2s
- configurations attempted: 122
- configurations with full LOSO + val: 122

## Top-15 configurations (by LOSO macro-F1)

| model      | feature_set      | preproc   | params_str                   |   loso_macro_f1_mean |   loso_macro_f1_std |   val_macro_f1 |   val_acc |
|:-----------|:-----------------|:----------|:-----------------------------|---------------------:|--------------------:|---------------:|----------:|
| lr_l2      | top80_cliff      | std       | {"C": 0.01}                  |                0.584 |               0.101 |          0.496 |     0.497 |
| lr_en      | top80_cliff      | std       | {"C": 0.1, "l1_ratio": 0.3}  |                0.583 |               0.094 |          0.517 |     0.517 |
| gnb        | top80_cliff      | std       | {}                           |                0.579 |               0.104 |          0.476 |     0.476 |
| lr_en      | top80_cliff      | std       | {"C": 0.1, "l1_ratio": 0.5}  |                0.577 |               0.098 |          0.506 |     0.507 |
| lr_en      | top80_cliff      | std       | {"C": 0.1, "l1_ratio": 0.7}  |                0.576 |               0.095 |          0.495 |     0.497 |
| lr_l1      | top80_cliff      | std       | {"C": 0.1}                   |                0.576 |               0.094 |          0.495 |     0.497 |
| lr_l2      | top40_cliff      | robust    | {"C": 1.0}                   |                0.575 |               0.101 |          0.500 |     0.500 |
| lr_l2      | top40_cliff      | std       | {"C": 1.0}                   |                0.573 |               0.098 |          0.500 |     0.500 |
| lr_l1      | top40_cliff      | robust    | {"C": 1.0}                   |                0.572 |               0.096 |          0.503 |     0.503 |
| lr_l1      | top40_cliff      | std       | {"C": 1.0}                   |                0.571 |               0.100 |          0.507 |     0.507 |
| lr_l1      | tight_pool_top40 | robust    | {"C": 1.0}                   |                0.570 |               0.095 |          0.510 |     0.510 |
| svm_rbf    | top80_cliff      | std       | {"C": 1.0, "gamma": "scale"} |                0.570 |               0.115 |          0.479 |     0.479 |
| lr_l1      | tight_pool_top40 | std       | {"C": 1.0}                   |                0.566 |               0.093 |          0.507 |     0.507 |
| svm_linear | top80_cliff      | std       | {"C": 0.1}                   |                0.563 |               0.109 |          0.510 |     0.510 |
| lr_l2      | tight_pool_top40 | robust    | {"C": 1.0}                   |                0.561 |               0.095 |          0.521 |     0.521 |

## Ensembles of top-K configs (mean predicted probability)

|      k |   loso_macro_f1 |   loso_acc |   loso_balanced_acc |   val_macro_f1 |   val_acc |   val_balanced_acc |
|-------:|----------------:|-----------:|--------------------:|---------------:|----------:|-------------------:|
|  3.000 |           0.584 |      0.584 |               0.584 |          0.469 |     0.469 |              0.469 |
|  5.000 |           0.574 |      0.574 |               0.574 |          0.462 |     0.462 |              0.462 |
| 10.000 |           0.582 |      0.582 |               0.582 |          0.479 |     0.479 |              0.479 |

## Notes
- Resumable: rerun the script to continue from the last fold.
- Per-fold trail: `results/tables/armhand_search_perfold.csv`.
- Top-K table: `results/tables/armhand_search_top.csv`.