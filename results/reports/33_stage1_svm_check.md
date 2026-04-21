# 33 — Stage-1 SVM Check

- runtime: 439.6s
- base search: 36 configs
- refinement search: 12 configs
- total configs: 48

## Best Overall

- config_id: truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base|anchor_z_l05
- resample_tag: truncate1022
- norm_id: subject_z
- model_id: svm_rbf_c3_gscale
- refine_id: anchor_z_l05
- macro_f1_mean: 0.8177083333333333
- accuracy_mean: 0.8379629629629629
- precision_nopain_mean: 0.7569444444444444
- recall_nopain_mean: 0.7569444444444444

## Top Validation Configs

| config | macro-F1 | acc | prec NoPain | rec NoPain |
|---|---:|---:|---:|---:|
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base|anchor_z_l05 | 0.818 | 0.838 | 0.757 | 0.757 |
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base|anchor_z_l05 | 0.818 | 0.838 | 0.757 | 0.757 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base|anchor_z_l05 | 0.818 | 0.838 | 0.757 | 0.757 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base|anchor_z_l05 | 0.818 | 0.838 | 0.757 | 0.757 |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base|anchor_z_l05 | 0.818 | 0.838 | 0.757 | 0.757 |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base|anchor_z_l05 | 0.818 | 0.838 | 0.757 | 0.757 |
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | 0.812 | 0.833 | 0.750 | 0.750 |
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | 0.812 | 0.833 | 0.750 | 0.750 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | 0.812 | 0.833 | 0.750 | 0.750 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | 0.812 | 0.833 | 0.750 | 0.750 |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | 0.812 | 0.833 | 0.750 | 0.750 |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | 0.812 | 0.833 | 0.750 | 0.750 |

## Base SVM Configs

| config | macro-F1 | acc | resample | norm | svm |
|---|---:|---:|---|---|---|
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | 0.812 | 0.833 | truncate1022 | subject_z | svm_rbf_c3_gscale |
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | 0.812 | 0.833 | truncate1022 | subject_z | svm_rbf_c3_g001 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | 0.812 | 0.833 | linear1022 | subject_z | svm_rbf_c3_gscale |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | 0.812 | 0.833 | linear1022 | subject_z | svm_rbf_c3_g001 |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | 0.812 | 0.833 | poly1022 | subject_z | svm_rbf_c3_gscale |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | 0.812 | 0.833 | poly1022 | subject_z | svm_rbf_c3_g001 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c1_gscale|base | 0.807 | 0.829 | linear1022 | subject_z | svm_rbf_c1_gscale |
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c1_gscale|base | 0.802 | 0.824 | truncate1022 | subject_z | svm_rbf_c1_gscale |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c1_gscale|base | 0.802 | 0.824 | poly1022 | subject_z | svm_rbf_c1_gscale |
| truncate1022|subject_z|bvp_eda_core|svm_linear_c03|base | 0.792 | 0.815 | truncate1022 | subject_z | svm_linear_c03 |
| linear1022|subject_z|bvp_eda_core|svm_linear_c03|base | 0.792 | 0.815 | linear1022 | subject_z | svm_linear_c03 |
| truncate1022|subject_robust|bvp_eda_core|svm_rbf_c1_gscale|base | 0.786 | 0.810 | truncate1022 | subject_robust | svm_rbf_c1_gscale |

## Best Anchor Refinements

| base | refine | macro-F1 | acc |
|---|---|---:|---:|
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | anchor_z_l05 | 0.818 | 0.838 |
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | anchor_z_l05 | 0.818 | 0.838 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | anchor_z_l05 | 0.818 | 0.838 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | anchor_z_l05 | 0.818 | 0.838 |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | anchor_z_l05 | 0.818 | 0.838 |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | anchor_z_l05 | 0.818 | 0.838 |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | anchor_center_l05 | 0.792 | 0.815 |
| poly1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | anchor_center_l05 | 0.792 | 0.815 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | anchor_center_l05 | 0.786 | 0.810 |
| linear1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | anchor_center_l05 | 0.786 | 0.810 |
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_gscale|base | anchor_center_l05 | 0.781 | 0.806 |
| truncate1022|subject_z|bvp_eda_core|svm_rbf_c3_g001|base | anchor_center_l05 | 0.781 | 0.806 |