# Tier A #1 -- Stimulus-onset alignment (ARM vs HAND)

Detection rate (train pain segments): **99.4%** (831/836)
Median onset ARM = 2.45 s,  HAND = 2.45 s

## Curated feature set (n = 22)

- Features tested: 22
- Features with smaller p after alignment: **8/22** (36%)
- Median p-ratio (aligned/unaligned): **1.478**
- Features FDR<0.05 unaligned: 0
- Features FDR<0.05 aligned:   0
- Features newly FDR-significant after alignment: 0 (none)
- Validation direction preserved on aligned features: **3/22** (14%)

## Top 10 features by |delta(Cliff's d)|

| feature | unaligned p | aligned p | p_ratio | delta_cliff |
|---|---|---|---|---|
| Bvp_kurtosis | 0.91 | 0.0945 | 0.104 | +0.028 |
| Bvp_petrosian_fd | 0.0269 | 0.0136 | 0.505 | +0.015 |
| Bvp_n_extrema | 0.0244 | 0.0131 | 0.54 | +0.011 |
| Eda_skew | 0.354 | 0.572 | 1.61 | +0.007 |
| Resp_iqr | 0.368 | 0.545 | 1.48 | -0.001 |
| Resp_range | 0.0892 | 0.0972 | 1.09 | -0.001 |
| Resp_amp_std | 0.452 | 0.942 | 2.08 | -0.002 |
| Resp_n_extrema | 0.327 | 0.283 | 0.864 | -0.005 |
| Eda_mean | 0.581 | 0.858 | 1.48 | -0.005 |
| Resp_mean_abs | 0.146 | 0.0918 | 0.629 | -0.006 |

## Verdict: **NOT compelling**

Rationale: alignment is judged useful if (a) it lifts at least one feature into FDR<0.05 that was not significant before, or (b) the median p-value reduces by 2x or more across the curated set.

Notes / caveats:
- Onset detection is heuristic. EDA slope, BVP amplitude drop, RESP perturbation each have ~30-60% individual hit rate; consensus + fallback to EDA-only when spread > 2 s mitigates outliers.
- Aligned window length is 6 s (vs unaligned 10 s); some loss of averaging power is expected. The comparison is therefore biased *against* alignment, so any improvement is meaningful.
