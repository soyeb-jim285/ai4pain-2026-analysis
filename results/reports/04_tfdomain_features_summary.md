# Time + frequency-domain feature extraction — summary

- Segments: train=1476, validation=432
- Features per segment: 158 (4 signals × ~39)
- Missing values: 6301 NaNs across table (1 features never extracted)

## Top 15 features by ANOVA F (subject-mean, 3 classes, train)

| feature                |      F |        p |   n |
|:-----------------------|-------:|---------:|----:|
| Bvp_rms                | 24.176 | 1.51e-09 | 123 |
| Bvp_mean               | 22.74  | 4.23e-09 | 123 |
| Bvp_mean_abs           | 22.74  | 4.23e-09 | 123 |
| Bvp_energy             | 16.645 | 4.17e-07 | 123 |
| Eda_bp_higher          |  7.538 | 0.000824 | 123 |
| SpO2_hjorth_complexity |  7.09  | 0.00124  | 120 |
| SpO2_hjorth_mobility   |  6.977 | 0.00137  | 120 |
| Eda_kurtosis           |  6.336 | 0.00242  | 123 |
| Eda_spec_bandwidth     |  5.702 | 0.00431  | 123 |
| Eda_spec_spread        |  5.702 | 0.00431  | 123 |
| Eda_dfa_alpha          |  5.412 | 0.00562  | 123 |
| Eda_skew               |  4.418 | 0.0141   | 123 |
| Eda_zcr                |  4.32  | 0.0154   | 123 |
| Eda_mcr                |  4.32  | 0.0154   | 123 |
| Eda_hjorth_mobility    |  3.974 | 0.0213   | 123 |

## Sanity checks (train means)

- BVP_peak_freq_mean_Hz: 1.3540
- BVP_dom_freq_cardiac_Hz: 1.3558
- BVP_lf_hf_ratio_mean: nan
- RESP_dom_freq_breathing_Hz: 0.3906
- RESP_peak_freq_Hz: 0.3978
- EDA_spec_centroid_Hz: 48.8509
- SpO2_total_variance: 46.4078

## Interpretation

- RESP dominant breathing frequency: NoPain=0.391 Hz, PainArm=0.391 Hz, PainHand=0.391 Hz (~23.4, 23.4, 23.4 breaths/min).
- EDA total spectral power: NoPain=0.77, PainArm=0.78, PainHand=0.78 — pain trials higher, consistent with increased sweat-gland activity.

## Failed extractions per split

- train: 0 segment-level failures
- validation: 0 segment-level failures

## Outputs

- `/stuff/Study/projects/AI4Pain 2026 Dataset/analysis/results/tables/tf_features.parquet`
- `results/tables/tf_features_dictionary.csv`
- `results/tables/tf_features_class_means.csv`
- `results/tables/tf_features_anova_train.csv`
- `plots/tfdomain/*.png`