# 10 Morphology + coupling features (PainArm vs PainHand)

- Total features: **52** (morphology + linear coupling + nonlinear coupling)
- Train subjects (paired Wilcoxon): 41
- FDR<0.05 within family: **0**
- FDR<0.05 globally (all features, single family): **0**

## Family breakdown

| family | n features | n FDR<0.05 | min p | median |delta| |
|--------|-----------:|-----------:|------:|---------------:|
| coupling_linear | 13 | 0 | 4.97e-02 | 0.065 |
| coupling_nonlinear | 12 | 0 | 1.47e-02 | 0.071 |
| morphology | 27 | 0 | 1.02e-02 | 0.033 |

## Top 15 features by raw Wilcoxon p

| rank | feature | family | W | p | p_fdr (family) | Cliff's d | sign | dir | val-dir-pres |
|-----:|---------|--------|---:|---:|---------------:|---------:|------:|-----|:------------:|
| 1 | `ppg_n_beats` | morphology | 219.00 | 1.02e-02 | 2.75e-01 | +0.044 | 0.65 | Arm > Hand | yes |
| 2 | `dcor_eda_spo2` | coupling_nonlinear | 230.00 | 1.47e-02 | 1.11e-01 | +0.151 | 0.68 | Arm > Hand | yes |
| 3 | `mi_eda_resp` | coupling_nonlinear | 250.00 | 1.85e-02 | 1.11e-01 | +0.137 | 0.63 | Arm > Hand | yes |
| 4 | `mi_bvp_eda` | coupling_nonlinear | 265.00 | 3.15e-02 | 1.26e-01 | +0.177 | 0.68 | Arm > Hand | yes |
| 5 | `ppg_halfwidth_std` | morphology | 267.00 | 3.37e-02 | 3.13e-01 | +0.053 | 0.63 | Arm > Hand | no |
| 6 | `ppg_fall_time_mean` | morphology | 268.00 | 3.48e-02 | 3.13e-01 | -0.032 | 0.66 | Arm < Hand | yes |
| 7 | `bvp_env_eda_xcorr_at_zero` | coupling_linear | 279.00 | 4.97e-02 | 6.46e-01 | +0.199 | 0.71 | Arm > Hand | yes |
| 8 | `dcor_eda_resp` | coupling_nonlinear | 280.00 | 5.12e-02 | 1.39e-01 | +0.145 | 0.66 | Arm > Hand | yes |
| 9 | `dcor_bvp_eda` | coupling_nonlinear | 284.00 | 5.80e-02 | 1.39e-01 | +0.168 | 0.63 | Arm > Hand | no |
| 10 | `ppg_rise_time_mean` | morphology | 295.00 | 8.02e-02 | 4.75e-01 | -0.061 | 0.63 | Arm < Hand | yes |
| 11 | `ppg_notch_time_frac_mean` | morphology | 307.00 | 1.12e-01 | 4.75e-01 | -0.049 | 0.61 | Arm < Hand | no |
| 12 | `ppg_sys_amp_mean` | morphology | 308.00 | 1.15e-01 | 4.75e-01 | -0.041 | 0.61 | Arm < Hand | yes |
| 13 | `ppg_aug_index_mean` | morphology | 316.00 | 1.41e-01 | 4.75e-01 | -0.035 | 0.61 | Arm < Hand | no |
| 14 | `ppg_notch_depth_rel_mean` | morphology | 316.00 | 1.41e-01 | 4.75e-01 | -0.035 | 0.61 | Arm < Hand | no |
| 15 | `bvp_env_eda_xcorr_max` | coupling_linear | 333.00 | 2.11e-01 | 7.67e-01 | +0.192 | 0.59 | Arm > Hand | yes |

## Did coupling outperform morphology?

- Morphology: min p = 1.02e-02, median |delta| = 0.033
- Coupling (linear+nonlinear): min p = 1.47e-02, median |delta| = 0.065
- Stronger family (by smallest p): **morphology**.

## Waveform morphology (notch depth / rise time / aug-index)

| feature | W | p | p_fdr (family) | delta | direction |
|---------|---:|---:|--------------:|------:|-----------|
| `ppg_notch_depth_rel_mean` | 316.00 | 1.41e-01 | 4.75e-01 | -0.035 | Arm < Hand |
| `ppg_notch_depth_rel_std` | 357.00 | 3.48e-01 | 7.12e-01 | +0.033 | Arm > Hand |
| `ppg_notch_time_frac_mean` | 307.00 | 1.12e-01 | 4.75e-01 | -0.049 | Arm < Hand |
| `ppg_rise_time_mean` | 295.00 | 8.02e-02 | 4.75e-01 | -0.061 | Arm < Hand |
| `ppg_rise_frac_mean` | 407.00 | 7.68e-01 | 9.59e-01 | -0.012 | Arm < Hand |
| `ppg_aug_index_mean` | 316.00 | 1.41e-01 | 4.75e-01 | -0.035 | Arm < Hand |
| `ppg_refl_index_mean` | 425.00 | 9.49e-01 | 9.59e-01 | -0.015 | Arm < Hand |
| `ppg_d2_peak_count_per_beat` | 335.00 | 2.21e-01 | 6.39e-01 | -0.026 | Arm < Hand |

## Physiological interpretation
- **PPG morphology** probes small-vessel compliance and reflected-wave timing. If PainHand (stimulated at the hand) produces stronger distal vasoconstriction than PainArm, expect shorter rise times, deeper dicrotic notch relative to systolic amplitude, and higher augmentation-index proxies on the hand condition.
- **RSA / RESP-BVP phase-locking** indexes parasympathetic modulation. A site-dependent shift in PLV or mean phase would suggest differential autonomic routing (afferent input from arm vs hand) even when segment-mean HR is matched.
- **BVP-envelope → EDA lag** captures the delay between pulse-amplitude responses and sudomotor activation. Hand stimulation might produce a shorter or larger-magnitude EDA response.
- **MI / dcor** catch nonlinear, non-phase-locked coupling (e.g. saturation, threshold effects) that linear r misses.

## Validation reproducibility (top 15)
- Direction preserved in validation: **10/15** (66.7%)

## Caveats
- Subject-z normalisation removes per-subject level effects; remaining class differences are shape/coupling-driven, not amplitude-driven.
- 10-s segments are short for respiratory-band PLV; PLV estimates are noisy per segment but the mean across 12 segments/class/subject is usable.
- FDR is applied within each feature family (morphology / linear coupling / nonlinear coupling) and also globally. Family-wise is the primary hypothesis.
- SpO2-involved pairs are skipped for constant-SpO2 segments (see inventory).