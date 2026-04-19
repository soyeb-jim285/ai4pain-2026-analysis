# Baseline classifiers summary (script 08)

## Headline numbers

- **Best 3-class LOSO macro-F1**: 0.575 ± 0.110  (model=logreg, preproc=subjectz, mean_acc=0.579)
- **Best binary LOSO macro-F1**: 0.819 ± 0.111  (model=xgb, preproc=subjectz, mean_acc=0.848)
- **Best arm-vs-hand LOSO macro-F1**: 0.559 ± 0.104  (model=logreg, preproc=subjectz, mean_acc=0.561)
- Validation macro-F1 for best 3-class config: **0.515**

Chance-level references: 3-class = 0.333, binary = 0.500, arm-vs-hand = 0.500.

## Does subject-z preprocessing help?

Mean macro-F1 across models by (label_scheme × preprocessing):

| label_scheme   |   global |   subjectz |
|:---------------|---------:|-----------:|
| 3class         |    0.433 |      0.557 |
| armhand        |    0.464 |      0.533 |
| binary         |    0.677 |      0.8   |

## Worst-performing subjects (bottom-5 LOSO accuracy, best 3-class config)

|   subject_held_out |   accuracy |   macro_f1 |
|-------------------:|-----------:|-----------:|
|                 46 |      0.25  |      0.25  |
|                 36 |      0.389 |      0.391 |
|                 56 |      0.389 |      0.392 |
|                 22 |      0.444 |      0.433 |
|                 55 |      0.444 |      0.437 |

## Top 15 features (blended importance, 3-class LOSO)

| feature               |   mean_rank |
|:----------------------|------------:|
| Eda_spec_bandwidth    |        7.83 |
| Bvp_diff_std          |       14.83 |
| Bvp_p90               |       16.5  |
| Bvp_peak_power        |       24.33 |
| Eda_dfa_alpha         |       27    |
| Bvp_total_power       |       29.33 |
| Eda_median            |       31.83 |
| Bvp_mean              |       34.17 |
| Eda_energy            |       35.17 |
| Bvp_energy            |       40    |
| Eda_hjorth_complexity |       40.33 |
| Eda_bp_higher_rel     |       40.33 |
| Eda_p90               |       40.5  |
| Eda_max               |       43.17 |
| Bvp_median            |       44    |

## Generalization gap (LOSO → validation)

| model   | preprocessing   | label_scheme   |   macro_f1_mean |   macro_f1 |   gap_loso_minus_val |
|:--------|:----------------|:---------------|----------------:|-----------:|---------------------:|
| logreg  | global          | 3class         |           0.417 |      0.482 |               -0.065 |
| rf      | global          | 3class         |           0.444 |      0.489 |               -0.045 |
| xgb     | global          | 3class         |           0.44  |      0.514 |               -0.074 |
| logreg  | subjectz        | 3class         |           0.575 |      0.515 |                0.06  |
| rf      | subjectz        | 3class         |           0.541 |      0.552 |               -0.011 |
| xgb     | subjectz        | 3class         |           0.556 |      0.533 |                0.023 |
| logreg  | global          | binary         |           0.659 |      0.723 |               -0.064 |
| rf      | global          | binary         |           0.684 |      0.736 |               -0.051 |
| xgb     | global          | binary         |           0.689 |      0.738 |               -0.049 |
| logreg  | subjectz        | binary         |           0.801 |      0.798 |                0.003 |
| rf      | subjectz        | binary         |           0.781 |      0.807 |               -0.026 |
| xgb     | subjectz        | binary         |           0.819 |      0.801 |                0.018 |
| logreg  | global          | armhand        |           0.479 |      0.527 |               -0.048 |
| rf      | global          | armhand        |           0.446 |      0.509 |               -0.063 |
| xgb     | global          | armhand        |           0.467 |      0.535 |               -0.068 |
| logreg  | subjectz        | armhand        |           0.559 |      0.51  |                0.049 |
| rf      | subjectz        | armhand        |           0.524 |      0.507 |                0.018 |
| xgb     | subjectz        | armhand        |           0.515 |      0.5   |                0.015 |

## Modelling recommendations

- Non-linear ensembles (RF / XGB) should outperform linear logreg when subject-z features carry the signal; compare above.
- Subject-z normalisation mitigates the known DC-offset confound — prefer it over global-z if the table above shows a gain.
- The bottom-5 LOSO subjects are candidates for per-subject artefact review before any heavier modelling.
- Large LOSO → validation gap means the training subjects are not representative; consider stratifying feature selection and using more conservative models.
- For the challenge: ensemble RF+XGB with subject-z features, tune on LOSO macro-F1; arm-vs-hand requires richer temporal features than aggregate stats offered here.