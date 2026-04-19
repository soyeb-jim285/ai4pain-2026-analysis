# Tier-C TVSymp on EDA — ARM vs HAND

- features: 12
- segments: 1908
- FDR<0.05 survivors: **0 / 12**
- LOSO macro-F1: **0.491 ± 0.098** (chance = 0.50)
- Validation macro-F1: **0.477**

Note: narrowband 0.08-0.24 Hz Butterworth approximates VFCDM used in paper 3. Less selective; feasibility-check only.

## All features by p-value

| feature                      |   n |         p |    mean_arm |   mean_hand |    p_fdr |
|:-----------------------------|----:|----------:|------------:|------------:|---------:|
| tvsymp_time_to_half_peak_s   |  41 | 0.0778666 |  0.00804878 |  0.00497967 | 0.679481 |
| tvsymp_early_minus_late      |  41 | 0.155533  |  2.21003    |  2.21845    | 0.679481 |
| tvsymp_peak_time_s           |  41 | 0.215978  |  1.2511     |  1.20157    | 0.679481 |
| tvsymp_early_mean_0_3s       |  41 | 0.230834  |  2.84574    |  2.85329    | 0.679481 |
| tvsymp_std                   |  41 | 0.317311  |  1          |  1          | 0.679481 |
| tvsymp_first_crossing_mean_s |  41 | 0.352964  |  0.00855691 |  0.00642276 | 0.679481 |
| tvsymp_rise_slope            |  41 | 0.396364  |  1.02517    |  1.02475    | 0.679481 |
| tvsymp_late_mean_7_10s       |  41 | 0.653485  |  0.635706   |  0.634841   | 0.921274 |
| tvsymp_centroid_s            |  41 | 0.690956  |  3.39895    |  3.39349    | 0.921274 |
| tvsymp_peak_amp              |  41 | 0.817507  |  3.40397    |  3.39944    | 0.979549 |
| tvsymp_mean                  |  41 | 0.969328  |  1.57349    |  1.57445    | 0.979549 |
| tvsymp_auc                   |  41 | 0.979549  | 15.7049     | 15.7144     | 0.979549 |
