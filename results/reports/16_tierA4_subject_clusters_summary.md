# Tier-A4: subject clustering for arm-vs-hand response types

- subjects (train): **41**
- usable features: **212**
- pooled paired Wilcoxon ARM vs HAND: **7** features p<0.01, **0** FDR<0.05

## Silhouette per (method, k)

| method | k | silhouette |
|---|---|---|
| agglo | 2 | 0.140 |
| agglo | 3 | 0.154 |
| agglo | 4 | 0.157 |
| agglo | 5 | 0.130 |
| kmeans | 2 | 0.147 |
| kmeans | 3 | 0.147 |
| kmeans | 4 | 0.164 |
| kmeans | 5 | 0.171 |

Best silhouette: **kmeans k=5 = 0.171**

## Pooled vs cluster-sum n_p<0.01 (uncorrected)

| scheme | cluster_sum n_p<0.01 | pooled | helps? |
|---|---|---|---|
| agglo_k2 | 19 | 7 | YES |
| agglo_k3 | 19 | 7 | YES |
| agglo_k4 | 15 | 7 | YES |
| agglo_k5 | 16 | 7 | YES |
| kmeans_k2 | 11 | 7 | YES |
| kmeans_k3 | 7 | 7 | no |
| kmeans_k4 | 4 | 7 | no |
| kmeans_k5 | 12 | 7 | YES |

Most promising scheme by this metric: **agglo_k2**

## Within-cluster LOSO macro-F1 (best scheme)

Pooled arm-vs-hand baseline (context): **0.54** macro-F1

| scheme | cluster | n_subj | n_seg | macro-F1 | acc | balanced acc |
|---|---|---|---|---|---|---|
| agglo_k2 | 0 | 23 | 552 | 0.544 | 0.547 | 0.547 |
| agglo_k2 | 1 | 18 | 432 | 0.500 | 0.505 | 0.505 |

**No cluster reaches LOSO macro-F1 > 0.60** across any (method, k).

## Top features distinguishing the 2-cluster solutions

**agglo_k2**:
- Bvp_spec_edge_95  (F=80.44, p=5.1e-11)
- Bvp_samp_entropy  (F=69.23, p=3.6e-10)
- Bvp_spec_bandwidth  (F=68.91, p=3.8e-10)
- Bvp_spec_spread  (F=68.91, p=3.8e-10)
- Bvp_hjorth_complexity  (F=68.45, p=4.1e-10)
- Bvp_approx_entropy  (F=68.10, p=4.4e-10)
- Bvp_hjorth_mobility  (F=49.70, p=1.8e-08)
- Bvp_n_extrema  (F=44.50, p=6.1e-08)
- Bvp_petrosian_fd  (F=44.33, p=6.3e-08)
- Bvp_p90  (F=42.56, p=9.7e-08)

**kmeans_k2**:
- Bvp_spec_edge_95  (F=160.87, p=2e-15)
- Bvp_spec_bandwidth  (F=111.43, p=5.4e-13)
- Bvp_spec_spread  (F=111.43, p=5.4e-13)
- Bvp_hjorth_complexity  (F=102.12, p=1.9e-12)
- Bvp_samp_entropy  (F=102.01, p=1.9e-12)
- Bvp_approx_entropy  (F=88.40, p=1.4e-11)
- Bvp_hjorth_mobility  (F=83.61, p=3e-11)
- Bvp_mcr  (F=71.47, p=2.4e-10)
- Bvp_zcr  (F=71.47, p=2.4e-10)
- Bvp_spec_entropy  (F=69.78, p=3.2e-10)

## Verdict
**Mixed signal.** Clustering increases the count of nominal-significant arm-vs-hand features above the pooled baseline, suggesting heterogeneity exists, but no single sub-group classifier crosses macro-F1 = 0.60. MoE is worth a small-scale prototype but not a major investment.

_runtime: 171.4s_