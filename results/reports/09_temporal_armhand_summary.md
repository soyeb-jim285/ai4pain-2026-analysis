# 09 Temporal dynamics: PainArm vs PainHand

Per-segment 10-s signal split into 5 non-overlapping 2-s windows. Per-window stats + trajectory summaries -> 440 temporal features across 4 signals. Paired within-subject tests on 41 train subjects (each subject -> one ARM score + one HAND score per feature).

## FDR-significant features
- Paired Wilcoxon (subject-mean ARM vs HAND), FDR q<0.05: **0** of 440.
- Mixed-effects LMM (segment-level, random subject intercept), FDR q<0.05: **4** of 440.

## Top 10 features by smallest paired-Wilcoxon p

| feature | signal | paired_p | paired_p_fdr | cliff_delta | sign_cons | lmm_coef | lmm_p | val_sign_preserved |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `Resp_w3_kurt` | Resp | 1.96e-03 | 2.86e-01 | -0.356 | 0.244 | 0.2909 | 1.34e-04 | yes |
| `Resp_w3_kurt__delta` | Resp | 1.96e-03 | 2.86e-01 | -0.325 | 0.244 | 0.2909 | 1.34e-04 | yes |
| `Resp_w3_skew` | Resp | 3.45e-03 | 2.86e-01 | -0.318 | 0.317 | 0.1729 | 3.16e-05 | yes |
| `Resp_w3_skew__delta` | Resp | 3.45e-03 | 2.86e-01 | -0.343 | 0.317 | 0.1729 | 3.16e-05 | yes |
| `Eda_w5_kurt` | Eda | 4.32e-03 | 2.86e-01 | 0.333 | 0.683 | -0.1088 | 1.09e-02 | yes |
| `Eda_w5_kurt__delta` | Eda | 4.32e-03 | 2.86e-01 | 0.337 | 0.683 | -0.1088 | 1.09e-02 | yes |
| `SpO2_w3_rms__delta` | SpO2 | 5.12e-03 | 2.91e-01 | -0.160 | 0.333 | 1.288 | 1.49e-01 | no |
| `SpO2_w3_rms` | SpO2 | 7.60e-03 | 3.78e-01 | -0.144 | 0.368 | 1.305 | 1.42e-01 | no |
| `Resp_w4_range__delta` | Resp | 2.39e-02 | 7.53e-01 | 0.135 | 0.683 | -0.01638 | 7.38e-02 | no |
| `Resp_w4_p2p` | Resp | 2.39e-02 | 7.53e-01 | 0.078 | 0.683 | -0.01638 | 7.38e-02 | no |

## Top 10 features by sign-consistency (|p - 0.5|)

| feature | signal | sign_cons | cliff_delta | paired_p | paired_p_fdr |
|---|---|---:|---:|---:|---:|
| `SpO2_slope_range__delta` | SpO2 | 1.000 | 1.000 | nan | nan |
| `SpO2_slope_range` | SpO2 | 1.000 | 1.000 | nan | nan |
| `Eda_w3_domfreq__delta` | Eda | 0.818 | 0.185 | 3.89e-02 | 7.77e-01 |
| `Eda_w3_domfreq` | Eda | 0.818 | 0.173 | 3.89e-02 | 7.77e-01 |
| `Resp_w3_kurt` | Resp | 0.244 | -0.356 | 1.96e-03 | 2.86e-01 |
| `Resp_w3_kurt__delta` | Resp | 0.244 | -0.325 | 1.96e-03 | 2.86e-01 |
| `SpO2_w4_mean` | SpO2 | 0.275 | -0.119 | 6.03e-02 | 9.02e-01 |
| `SpO2_w4_mean__delta` | SpO2 | 0.282 | -0.080 | 8.48e-02 | 9.02e-01 |
| `SpO2_w5_kurt` | SpO2 | 0.692 | 0.252 | 3.86e-02 | 7.77e-01 |
| `Resp_w1_domfreq` | Resp | 0.692 | 0.057 | 1.57e-01 | 9.02e-01 |

## Within-subject discriminability
- Subjects with evaluable LOO CV: **41**.
- Median per-subject LOO accuracy (ARM vs HAND): **0.500** (chance = 0.5).
- Subjects with LOO > 0.7 ('discriminable'): **1**.

Discriminable subjects:

| subject | n_segments | LOO accuracy |
|---:|---:|---:|
| 29 | 24 | 0.750 |

## Recommendation
- Temporal structure does not materially improve global ARM-vs-HAND separability: 0 features at FDR, and only 1 subjects are personally discriminable above 0.7. This dataset likely lacks a systematic, subject-invariant arm-vs-hand fingerprint in the 4 physiological channels at 100 Hz. Any usable separation would require either richer morphology features, subject calibration, or accepting near-chance performance.

## Methodological notes
- SpO2 flatline segments (std < 1e-6) are set to NaN for all SpO2 features.
- Subject-z delta features are computed as (feature - subject's NoPain mean) using both train and validation NoPain segments per subject.
- Paired Wilcoxon operates on subject-means (one ARM score + one HAND score per subject). Mixed LMM uses all ARM+HAND segments with subject random intercepts.
- FDR (BH, alpha=0.05) applied separately for paired-Wilcoxon and LMM families.
- Validation direction-preservation is a coarse sanity check using sign of ARM-HAND difference on the validation split.
