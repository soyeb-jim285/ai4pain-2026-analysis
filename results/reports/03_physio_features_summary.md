# 03 Physiological Features Summary

Segments processed: 1908 (1476 train + 432 validation). Sampling rate: 100 Hz.

## Extraction reliability

Status is tracked per representative feature (other features in the same signal share the same extraction path and inherit the outcome for their segment). `ok` = main method, `fallback` = manual backup, `nan` = skipped (segment too short or error).

| feature | ok | fallback | nan | % success |
|---|---:|---:|---:|---:|
| BVP_hr_mean | 1908 | 0 | 0 | 100.0% |
| BVP_peak_count | 0 | 0 | 0 | nan% |
| BVP_rr_mean | 0 | 0 | 0 | nan% |
| BVP_sdnn | 0 | 0 | 0 | nan% |
| BVP_rmssd | 0 | 0 | 0 | nan% |
| BVP_pnn20 | 0 | 0 | 0 | nan% |
| BVP_peak_amp_mean | 0 | 0 | 0 | nan% |
| BVP_peak_amp_std | 0 | 0 | 0 | nan% |
| BVP_risetime_mean | 0 | 0 | 0 | nan% |
| BVP_env_std | 0 | 0 | 0 | nan% |
| EDA_tonic_mean | 1908 | 0 | 0 | 100.0% |
| EDA_tonic_slope | 0 | 0 | 0 | nan% |
| EDA_scr_count | 1908 | 0 | 0 | 100.0% |
| EDA_scr_amp_mean | 0 | 0 | 0 | nan% |
| EDA_scr_rate | 0 | 0 | 0 | nan% |
| EDA_phasic_auc | 0 | 0 | 0 | nan% |
| EDA_range | 0 | 0 | 0 | nan% |
| RESP_rate | 1908 | 0 | 0 | 100.0% |
| RESP_interval_std | 0 | 0 | 0 | nan% |
| RESP_amp_mean | 0 | 0 | 0 | nan% |
| RESP_amp_std | 0 | 0 | 0 | nan% |
| RESP_ie_ratio | 0 | 0 | 0 | nan% |
| SPO2_mean | 1908 | 0 | 0 | 100.0% |
| SPO2_std | 0 | 0 | 0 | nan% |
| SPO2_min | 0 | 0 | 0 | nan% |
| SPO2_dip_magnitude | 0 | 0 | 0 | nan% |
| SPO2_dip_count | 0 | 0 | 0 | nan% |
| SPO2_slope | 0 | 0 | 0 | nan% |
| BVP_RESP_envcorr | 1908 | 0 | 0 | 100.0% |

## Top 10 class-discriminative features (train, subject-averaged ANOVA)

Computed on subject-level means (one row per (subject, class)) to reduce pseudoreplication.

| rank | feature | F | p | NoPain | PainArm | PainHand |
|---:|---|---:|---:|---:|---:|---:|
| 1 | EDA_tonic_slope | 5.506 | 5.16e-03 | -0.011 | 0.020 | 0.025 |
| 2 | EDA_range | 2.088 | 1.28e-01 | 2.598 | 3.037 | 3.061 |
| 3 | EDA_tonic_mean | 1.593 | 2.08e-01 | 3.649 | 4.527 | 4.571 |
| 4 | RESP_rate | 1.489 | 2.30e-01 | 16.890 | 17.829 | 17.372 |
| 5 | BVP_peak_amp_mean | 1.422 | 2.45e-01 | 0.013 | 0.010 | 0.010 |
| 6 | EDA_scr_count | 1.320 | 2.71e-01 | 9.490 | 9.421 | 9.478 |
| 7 | EDA_scr_rate | 1.307 | 2.74e-01 | 9.492 | 9.423 | 9.480 |
| 8 | SPO2_dip_count | 1.138 | 3.24e-01 | 0.124 | 0.169 | 0.173 |
| 9 | SPO2_std | 0.908 | 4.06e-01 | 2.764 | 3.572 | 3.677 |
| 10 | BVP_env_std | 0.866 | 4.23e-01 | 0.004 | 0.003 | 0.003 |

## Physiological interpretation

- **EDA_tonic_slope** (F=5.51, p=5.2e-03): Pain higher than NoPain (NoPain=-0.011, PainArm=0.020, PainHand=0.025). (EDA rises with sympathetic activation)
- **EDA_range** (F=2.09, p=1.3e-01): Pain higher than NoPain (NoPain=2.598, PainArm=3.037, PainHand=3.061). (EDA rises with sympathetic activation)
- **EDA_tonic_mean** (F=1.59, p=2.1e-01): Pain higher than NoPain (NoPain=3.649, PainArm=4.527, PainHand=4.571). (EDA rises with sympathetic activation)
- **RESP_rate** (F=1.49, p=2.3e-01): Pain higher than NoPain (NoPain=16.890, PainArm=17.829, PainHand=17.372). (breathing rate and variability shift with pain/stress)
- **BVP_peak_amp_mean** (F=1.42, p=2.5e-01): Pain lower than NoPain (NoPain=0.013, PainArm=0.010, PainHand=0.010).
- **EDA_scr_count** (F=1.32, p=2.7e-01): Pain lower than NoPain (NoPain=9.490, PainArm=9.421, PainHand=9.478). (SCR activity reflects sympathetic arousal)
- **EDA_scr_rate** (F=1.31, p=2.7e-01): Pain lower than NoPain (NoPain=9.492, PainArm=9.423, PainHand=9.480). (SCR activity reflects sympathetic arousal)
- **SPO2_dip_count** (F=1.14, p=3.2e-01): Pain higher than NoPain (NoPain=0.124, PainArm=0.169, PainHand=0.173). (peripheral vasoconstriction under pain can shift SpO2)
- **SPO2_std** (F=0.91, p=4.1e-01): Pain higher than NoPain (NoPain=2.764, PainArm=3.572, PainHand=3.677). (peripheral vasoconstriction under pain can shift SpO2)
- **BVP_env_std** (F=0.87, p=4.2e-01): Pain lower than NoPain (NoPain=0.004, PainArm=0.003, PainHand=0.003).

## Sanity checks

- HR median=78.7 bpm (P5=60.0, P95=101.5). OK
- RESP median=17.4 /min (P5=10.7, P95=24.2). OK
- SpO2 median=93.3% (P5=42.1, P95=100.0). OK

## Outputs

- `results/tables/physio_features.parquet`
- `results/tables/physio_features_dictionary.csv`
- `results/tables/physio_features_class_means.csv`
- `results/tables/physio_features_extraction_status.csv`
- `plots/physio/feature_boxplots_by_class.png`
- `plots/physio/bvp_hr_per_class.png`
- `plots/physio/eda_scr_count_per_class.png`
- `plots/physio/resp_rate_per_class.png`
- `plots/physio/spo2_mean_per_class.png`
- `plots/physio/example_subject_{sid}_hr_over_segments.png` (x6)