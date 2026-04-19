# 11 - Reactivity + Interactions: ARM vs HAND

## Features per family

- **reactivity**: tested=510, FDR<0.05=0
- **interaction**: tested=50, FDR<0.05=0
- **multiscale**: tested=86, FDR<0.05=0

## Top 15 features across families (by smallest p)

| feature | family | W | p | p_fdr | Cliff | sign_cons | direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ratio_Resp_skew | reactivity | 203.0 | 0.00261 | 0.277 | -0.234 | 0.73 | HAND > ARM |
| ratio_raw_Resp_skew | reactivity | 203.0 | 0.00261 | 0.277 | -0.234 | 0.73 | HAND > ARM |
| delta_RESP_amp_std | reactivity | 204.0 | 0.00274 | 0.277 | 0.191 | 0.63 | ARM > HAND |
| delta_ms_pow_Resp_b1_0.2-0.5 | multiscale | 215.0 | 0.00451 | 0.277 | 0.072 | 0.66 | ARM > HAND |
| delta_raw_Resp_mad | reactivity | 218.0 | 0.00515 | 0.277 | 0.152 | 0.71 | ARM > HAND |
| delta_Resp_mad | reactivity | 218.0 | 0.00515 | 0.277 | 0.152 | 0.71 | ARM > HAND |
| zdev_Resp_mad | reactivity | 218.0 | 0.00515 | 0.277 | 0.111 | 0.71 | ARM > HAND |
| zdev_raw_Resp_mad | reactivity | 218.0 | 0.00515 | 0.277 | 0.111 | 0.71 | ARM > HAND |
| ratio_RESP_amp_std | reactivity | 219.0 | 0.00537 | 0.277 | 0.154 | 0.63 | ARM > HAND |
| delta_ms_std_Resp_b1_0.2-0.5 | multiscale | 223.0 | 0.00638 | 0.277 | 0.065 | 0.68 | ARM > HAND |
| ratdd_delta_Resp_median__delta_raw_Bvp_mad | interaction | 224.0 | 0.00665 | 0.277 | -0.315 | 0.71 | HAND > ARM |
| delta_Resp_diff_std | reactivity | 224.0 | 0.00665 | 0.277 | 0.143 | 0.66 | ARM > HAND |
| ratdd_delta_Resp_median__delta_Bvp_mad | interaction | 224.0 | 0.00665 | 0.277 | -0.315 | 0.71 | HAND > ARM |
| ratio_Resp_mad | reactivity | 227.0 | 0.00754 | 0.277 | 0.142 | 0.71 | ARM > HAND |
| ratio_raw_Resp_mad | reactivity | 227.0 | 0.00754 | 0.277 | 0.142 | 0.71 | ARM > HAND |

## Reactivity vs raw form

- % feature pairs where reactivity p < raw p: **29.6%**

## Location-discriminable subjects

- n discriminable (>=10 features |effect|>0.5): **41/41**
- distribution of #features with |std effect|>0.5 per subject: min=30, Q1=75, median=105, Q3=147, max=219
- Note: with ~510 reactivity features the |effect|>0.5 threshold is not a strong criterion; all 41 subjects exceed 10. The distribution width is the useful signal — high-end subjects show several-fold more arm-vs-hand separability than low-end subjects.

## Top interaction feature

- **ratdd_delta_Resp_median__delta_raw_Bvp_mad**: W=224.0, p=0.00665, p_fdr=0.277, Cliff=-0.315, direction=HAND > ARM

## Recommendation

- **No feature survives FDR<0.05** across reactivity / interaction / multiscale families. Within-subject baseline normalisation does not expose a robust ARM vs HAND signal.
- Conclusion: arm-vs-hand appears genuinely subject-specific; a small sub-group of subjects (see strata file) may carry the effect. Consider subject-mixture or attention-based models over raw time series rather than aggregate features.