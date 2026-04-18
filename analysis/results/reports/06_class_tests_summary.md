# 06 Class-separability tests (subject-mean)

Per-feature omnibus + pairwise tests on subject-mean feature values (41 train subjects x 3 classes = 123 data points per feature). Tests on subject means avoid pseudoreplication caused by the dominant subject effect.

## 3-class ANOVA (FDR < 0.05)
- Total features tested: **243**
- Significant at Benjamini-Hochberg FDR q<0.05: **11** (4.5%)
- Breakdown by physiological axis:
  - BVP/HR: 6
  - EDA: 3
  - SpO2: 2

## Top 20 features by ANOVA F

| rank | feature | source | F | eta^2 | FDR p | NoPain | PainArm | PainHand | direction |
|-----:|---------|--------|---:|------:|------:|-------:|--------:|---------:|-----------|
| 1 | `raw_Bvp_rms` | raw | 24.18 | 0.287 | 1.69e-07 | 0.499 | 0.498 | 0.498 | Pain < NoPain (NoPain > PainArm > PainHand) |
| 2 | `Bvp_rms` | tf | 24.18 | 0.287 | 1.69e-07 | 0.499 | 0.498 | 0.498 | Pain < NoPain (NoPain > PainArm > PainHand) |
| 3 | `raw_Bvp_mean` | raw | 22.74 | 0.275 | 1.90e-07 | 0.499 | 0.498 | 0.498 | Pain < NoPain (NoPain > PainArm > PainHand) |
| 4 | `Bvp_mean` | tf | 22.74 | 0.275 | 1.90e-07 | 0.499 | 0.498 | 0.498 | Pain < NoPain (NoPain > PainArm > PainHand) |
| 5 | `Bvp_mean_abs` | tf | 22.74 | 0.275 | 1.90e-07 | 0.499 | 0.498 | 0.498 | Pain < NoPain (NoPain > PainArm > PainHand) |
| 6 | `Bvp_energy` | tf | 16.65 | 0.217 | 1.56e-05 | 249 | 248 | 248 | Pain < NoPain (NoPain > PainArm > PainHand) |
| 7 | `Eda_bp_higher` | tf | 7.54 | 0.112 | 2.65e-02 | 0.00013 | 0.000382 | 0.000402 | Pain > NoPain (PainHand > PainArm > NoPain) |
| 8 | `SpO2_hjorth_complexity` | tf | 7.09 | 0.108 | 3.43e-02 | 16 | 17.9 | 17.8 | Pain > NoPain (PainArm > PainHand > NoPain) |
| 9 | `SpO2_hjorth_mobility` | tf | 6.98 | 0.107 | 3.43e-02 | 0.0954 | 0.0863 | 0.0861 | Pain < NoPain (NoPain > PainArm > PainHand) |
| 10 | `Eda_kurtosis` | tf | 6.34 | 0.096 | 4.95e-02 | -1.67 | -1.44 | -1.46 | Pain > NoPain (PainArm > PainHand > NoPain) |
| 11 | `raw_Eda_kurtosis` | raw | 6.34 | 0.096 | 4.95e-02 | -1.67 | -1.44 | -1.46 | Pain > NoPain (PainArm > PainHand > NoPain) |

## Pain detection: Pain (Arm+Hand pooled) vs NoPain
- Features with FDR-adjusted ANOVA p<0.05: **17** of 243.
- Median |Cliff's delta| among top 10 by F: **0.694**.

Top 10 features (by ANOVA F):

| feature | source | F | FDR p | Cliff's delta | direction |
|---------|--------|---:|------:|-------------:|-----------|
| `raw_Bvp_rms` | raw | 49.11 | 7.71e-08 | -0.716 | Pain < NoPain |
| `Bvp_rms` | tf | 49.11 | 7.71e-08 | -0.716 | Pain < NoPain |
| `raw_Bvp_mean` | raw | 46.17 | 7.89e-08 | -0.710 | Pain < NoPain |
| `Bvp_mean_abs` | tf | 46.17 | 7.89e-08 | -0.710 | Pain < NoPain |
| `Bvp_mean` | tf | 46.17 | 7.89e-08 | -0.710 | Pain < NoPain |
| `Bvp_energy` | tf | 30.51 | 1.51e-05 | -0.678 | Pain < NoPain |
| `Eda_bp_higher` | tf | 17.50 | 2.35e-03 | 0.578 | Pain > NoPain |
| `SpO2_hjorth_complexity` | tf | 12.87 | 1.45e-02 | 0.484 | Pain > NoPain |
| `EDA_tonic_slope` | physio | 12.86 | 1.45e-02 | 0.480 | Pain > NoPain |
| `Eda_kurtosis` | tf | 12.06 | 1.70e-02 | 0.530 | Pain > NoPain |

## Pain localisation: PainArm vs PainHand (paired Wilcoxon)
- Features with FDR-adjusted paired-Wilcoxon p<0.05: **0** of 243.
- No features reach FDR<0.05 at the subject-mean level. Arm vs Hand is intrinsically the harder problem here (same stimulus, different site), and the subject-mean aggregation throws away within-subject segment-level variance — expected to be weak.

Top 10 features by smallest paired-Wilcoxon FDR p:

| feature | source | Wilcoxon FDR p | rank-biserial | Cliff's delta | direction |
|---------|--------|--------------:|-------------:|-------------:|-----------|
| `RESP_amp_std` | physio | 2.74e-01 | 0.526 | 0.135 | Arm > Hand |
| `Resp_mad` | tf | 2.74e-01 | 0.494 | 0.070 | Arm > Hand |
| `raw_Resp_mad` | raw | 2.74e-01 | 0.494 | 0.070 | Arm > Hand |
| `Resp_diff_std` | raw | 2.74e-01 | 0.480 | 0.080 | Arm > Hand |
| `SPO2_dip_magnitude` | physio | 2.74e-01 | -0.480 | -0.090 | Arm < Hand |
| `raw_Resp_skew` | raw | 2.74e-01 | -0.466 | -0.149 | Arm < Hand |
| `Resp_skew` | tf | 2.74e-01 | -0.466 | -0.149 | Arm < Hand |
| `Bvp_petrosian_fd` | tf | 2.98e-01 | 0.447 | 0.079 | Arm > Hand |
| `Bvp_n_extrema` | raw | 2.98e-01 | 0.446 | 0.071 | Arm > Hand |
| `SpO2_energy` | tf | 2.98e-01 | -0.449 | -0.068 | Arm < Hand |

## Validation-split consistency (top 30 train ANOVA features)
- Direction preserved in validation for **11/11** features (100.0%).

## Plain-English interpretation
- The physiological axis carrying the most class information is **BVP/HR**, followed by EDA (3), SpO2 (2).
- Pain vs NoPain is clearly detectable; Arm vs Hand is much subtler.
- Direction preservation on the validation split indicates whether the per-feature class shift generalises to held-out subjects.

## Methodological caveats
- Subject-mean aggregation collapses 12 segments per subject per class into a single data point. This correctly controls for the subject effect but deliberately discards within-subject variance, making per-feature tests conservative relative to segment-level pooling.
- Mann-Whitney U on subject means is unpaired; the paired Wilcoxon exploits the within-subject design and is the primary test for each pairwise contrast.
- Benjamini-Hochberg FDR is applied *within* each test family (ANOVA, Kruskal-Wallis, each pairwise Mann-Whitney, each pairwise Wilcoxon, Pain-vs-NoPain, Arm-vs-Hand). No correction across families.
- Cliff's delta values are computed on the unpaired subject-mean arrays; rank-biserial correlations use the paired-Wilcoxon signed ranks.
- Validation split has many fewer subjects; small per-class N on that split limits the precision of the val means. Direction agreement is a weak proxy for generalisation but good for flagging obviously unstable features.
