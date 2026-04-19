# 20 — Tier-B waveform residual + filter sweep

## Q1 — per-sample paired Wilcoxon (subject-residual)

Counts of samples with p<0.05 / FDR<0.05 across the 1000-sample window:

| derivative | signal | n_p<0.05 | n_FDR<0.05 |
|---|---|---|---|
| ddx | Bvp | 48 | 0 |
| ddx | Eda | 33 | 0 |
| ddx | Resp | 59 | 0 |
| ddx | SpO2 | 2 | 0 |
| dx | Bvp | 45 | 0 |
| dx | Eda | 31 | 0 |
| dx | Resp | 46 | 0 |
| dx | SpO2 | 1 | 0 |
| x | Bvp | 82 | 0 |
| x | Eda | 0 | 0 |
| x | Resp | 16 | 0 |
| x | SpO2 | 63 | 0 |

## Q2 — aggregate dx/ddx features (paired Wilcoxon FDR)

- 6/56 features survive BH-FDR<0.05
- 6/56 have raw p<0.01

Top 10 by smallest p:

| feature       |      p |   p_fdr |   cliff |   sign_arm_gt_hand |   mean_arm |   mean_hand |
|:--------------|-------:|--------:|--------:|-------------------:|-----------:|------------:|
| Resp_d2_range | 0.0001 |  0.0039 |  0.0696 |             0.7805 |     0.0095 |      0.0091 |
| Resp_d1_rms   | 0.0007 |  0.0139 |  0.0886 |             0.7561 |     0.0033 |      0.0031 |
| Resp_d1_std   | 0.0007 |  0.0139 |  0.0898 |             0.7561 |     0.0033 |      0.0031 |
| Resp_d1_range | 0.0016 |  0.0226 |  0.1017 |             0.6829 |     0.0187 |      0.0174 |
| Bvp_d2_std    | 0.0038 |  0.0352 |  0.0494 |             0.5854 |     0.0045 |      0.0045 |
| Bvp_d2_rms    | 0.0038 |  0.0352 |  0.0494 |             0.5854 |     0.0045 |      0.0045 |
| Bvp_d1_zcr    | 0.0258 |  0.1958 |  0.0410 |             0.6341 |     0.6227 |      0.6202 |
| Bvp_d1_rms    | 0.0315 |  0.1958 |  0.0410 |             0.6098 |     0.0028 |      0.0027 |
| Bvp_d1_std    | 0.0315 |  0.1958 |  0.0422 |             0.6098 |     0.0028 |      0.0027 |
| SpO2_d2_range | 0.1127 |  0.5651 | -0.0369 |             0.2927 |    17.7589 |     18.6813 |

## Q3 — filter sweep (per-channel LR LOSO on flattened residual)

Chance = 0.50.

| filter     | signal   |   loso_macro_f1_mean |   loso_macro_f1_std |   n_samples_per_seg |
|:-----------|:---------|---------------------:|--------------------:|--------------------:|
| bp_0.05_1  | Eda      |                0.507 |               0.084 |                 250 |
| detrend    | Bvp      |                0.503 |               0.106 |                 250 |
| bp_0.1_0.5 | Resp     |                0.501 |               0.112 |                 250 |
| none       | Bvp      |                0.498 |               0.108 |                 250 |
| savgol_25  | Bvp      |                0.494 |               0.099 |                 250 |
| bp_0.1_0.5 | SpO2     |                0.487 |               0.115 |                 250 |
| bp_0.1_0.5 | Eda      |                0.485 |               0.070 |                 250 |
| none       | Resp     |                0.483 |               0.116 |                 250 |
| detrend    | Resp     |                0.483 |               0.102 |                 250 |
| bp_0.05_1  | Resp     |                0.481 |               0.111 |                 250 |
| none       | Eda      |                0.480 |               0.110 |                 250 |
| savgol_25  | Eda      |                0.480 |               0.091 |                 250 |
| bp_0.05_1  | SpO2     |                0.480 |               0.113 |                 250 |
| savgol_25  | Resp     |                0.478 |               0.111 |                 250 |
| bp_0.05_1  | Bvp      |                0.472 |               0.094 |                 250 |
| bp_0.1_0.5 | Bvp      |                0.470 |               0.094 |                 250 |
| savgol_25  | SpO2     |                0.470 |               0.093 |                 250 |
| none       | SpO2     |                0.467 |               0.088 |                 250 |
| detrend    | Eda      |                0.465 |               0.103 |                 250 |
| detrend    | SpO2     |                0.464 |               0.093 |                 250 |

## Outputs

- `results/tables/tierB_residual_per_sample_pvals.parquet`
- `results/tables/tierB_derivative_features.parquet`
- `results/tables/tierB_derivative_tests.csv`
- `results/tables/tierB_filter_sweep.csv`
- `plots/tierB_waveform/{mean_residual_per_channel_*,sample_signif_heatmap,filter_sweep_bar}.png`