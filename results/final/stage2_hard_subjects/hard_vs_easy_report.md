# Hard-Subject Diagnosis

- k = 10 (hard = bottom-k by LOSO macro-F1, easy = top-k)
- hard subjects: [55, 46, 56, 15, 45, 25, 22, 11, 4, 63]
- easy subjects: [28, 48, 6, 43, 53, 1, 5, 47, 34, 44]
- hard mean F1: 0.4083
- easy mean F1: 0.6917

## Group comparison

| metric            |   hard_mean |   hard_median |   easy_mean |   easy_median |   delta_mean |   cliffs_delta_easy_vs_hard |
|:------------------|------------:|--------------:|------------:|--------------:|-------------:|----------------------------:|
| reactivity_l2     | 178370.9956 |    94030.8789 | 572711.7488 |   444183.3438 |  394340.7532 |                      0.5600 |
| reactivity_mean   |    868.1158 |      477.8777 |   2785.6367 |     2206.3999 |    1917.5209 |                      0.5600 |
| arm_hand_sep_mean |      0.1978 |        0.1744 |      0.2052 |        0.2101 |       0.0075 |                      0.1000 |
| arm_hand_sep_max  |      0.5417 |        0.5556 |      0.5410 |        0.5208 |      -0.0007 |                     -0.1100 |
| nan_frac          |      0.0000 |        0.0000 |      0.0000 |        0.0000 |       0.0000 |                      0.0000 |
| within_std_mean   |   3137.4565 |     2423.8015 |   3595.7615 |     3204.2413 |     458.3049 |                      0.0200 |

## Metric ↔ LOSO F1 correlation

| metric            |   pearson_r_vs_loso_f1 |   n |
|:------------------|-----------------------:|----:|
| reactivity_mean   |                 0.3799 |  41 |
| reactivity_l2     |                 0.3759 |  41 |
| within_std_mean   |                 0.1495 |  41 |
| arm_hand_sep_max  |                 0.0662 |  41 |
| arm_hand_sep_mean |                 0.0251 |  41 |
| nan_frac          |               nan      |  41 |

## Per-modality arm-vs-hand separability

| group   | modality   |   mean |   median |   count |
|:--------|:-----------|-------:|---------:|--------:|
| easy    | bvp        | 0.1983 |   0.1755 |      10 |
| easy    | eda        | 0.1909 |   0.1890 |      10 |
| easy    | resp       | 0.2052 |   0.2101 |      10 |
| hard    | bvp        | 0.1588 |   0.1375 |      10 |
| hard    | eda        | 0.1751 |   0.1808 |      10 |
| hard    | resp       | 0.1978 |   0.1744 |      10 |
| mid     | bvp        | 0.1990 |   0.1818 |      33 |
| mid     | eda        | 0.1802 |   0.1573 |      33 |
| mid     | resp       | 0.1975 |   0.1795 |      33 |

## Interpretation guide

- High `|r|` with LOSO F1 + clear hard/easy gap = that metric drives poor performance.
- If `arm_hand_sep_mean` is the top driver: hard subjects genuinely lack Arm-vs-Hand signal.
- If `reactivity_l2` dominates: hard subjects are weak responders (low NoPain->Pain delta).
- If `nan_frac` or `within_std_mean` dominate: signal-quality / sensor issue.
- If no single metric explains it: mixed causes, may need per-subject modelling.