# 32 — Stage-1 Upgrade Suite

- runtime: 0.0s
- base search: 72 configs
- refinement search: 24 configs
- total full pipeline configs: 96

## Best Overall

- config_id: truncate1022|subject_robust|bvp_eda_core|xgb|base|anchor_center_l05
- resample_tag: truncate1022
- norm_id: subject_robust
- feature_set: bvp_eda_core
- model_id: xgb
- refine_id: anchor_center_l05
- macro_f1_mean: 0.8333333333333333
- accuracy_mean: 0.8518518518518519
- precision_nopain_mean: 0.7777777777777778
- recall_nopain_mean: 0.7777777777777778

## Top Validation Configs

| config | macro-F1 | acc | prec NoPain | rec NoPain |
|---|---:|---:|---:|---:|
| truncate1022|subject_robust|bvp_eda_core|xgb|base|anchor_center_l05 | 0.833 | 0.852 | 0.778 | 0.778 |
| truncate1022|subject_robust|bvp_eda_core|xgb|base | 0.828 | 0.847 | 0.771 | 0.771 |
| poly1022|subject_robust|bvp_eda_core|xgb|base | 0.828 | 0.847 | 0.771 | 0.771 |
| truncate1022|subject_robust|bvp_eda_core|xgb|base|anchor_z_l05 | 0.828 | 0.847 | 0.771 | 0.771 |
| poly1022|subject_robust|bvp_eda_core|xgb|base|anchor_z_l05 | 0.828 | 0.847 | 0.771 | 0.771 |
| truncate1022|subject_z|bvp_eda_core|xgb|base | 0.823 | 0.843 | 0.764 | 0.764 |
| linear1022|subject_robust|bvp_eda_core|xgb|base | 0.823 | 0.843 | 0.764 | 0.764 |
| truncate1022|subject_robust|bvp_eda_core|xgb|base|anchor_z_l10 | 0.823 | 0.843 | 0.764 | 0.764 |
| poly1022|subject_robust|bvp_eda_core|xgb|base|anchor_center_l05 | 0.823 | 0.843 | 0.764 | 0.764 |
| truncate1022|subject_z|bvp_eda_core|xgb|base|anchor_z_l05 | 0.823 | 0.843 | 0.764 | 0.764 |
| truncate1022|subject_z|bvp_eda_core|xgb|base|anchor_z_l10 | 0.823 | 0.843 | 0.764 | 0.764 |
| linear1022|subject_robust|bvp_eda_core|xgb|base|anchor_center_l05 | 0.823 | 0.843 | 0.764 | 0.764 |

## Best Base Configs

| config | macro-F1 | acc | resample | norm | features | model |
|---|---:|---:|---|---|---|---|
| truncate1022|subject_robust|bvp_eda_core|xgb|base | 0.828 | 0.847 | truncate1022 | subject_robust | bvp_eda_core | xgb |
| poly1022|subject_robust|bvp_eda_core|xgb|base | 0.828 | 0.847 | poly1022 | subject_robust | bvp_eda_core | xgb |
| truncate1022|subject_z|bvp_eda_core|xgb|base | 0.823 | 0.843 | truncate1022 | subject_z | bvp_eda_core | xgb |
| linear1022|subject_robust|bvp_eda_core|xgb|base | 0.823 | 0.843 | linear1022 | subject_robust | bvp_eda_core | xgb |
| truncate1022|subject_robust|bvp_eda_resp_small|xgb|base | 0.818 | 0.838 | truncate1022 | subject_robust | bvp_eda_resp_small | xgb |
| linear1022|subject_z|bvp_eda_resp_small|xgb|base | 0.818 | 0.838 | linear1022 | subject_z | bvp_eda_resp_small | xgb |
| poly1022|subject_robust|bvp_eda_resp_small|xgb|base | 0.818 | 0.838 | poly1022 | subject_robust | bvp_eda_resp_small | xgb |
| poly1022|subject_z|bvp_eda_resp_small|rf|base | 0.812 | 0.833 | poly1022 | subject_z | bvp_eda_resp_small | rf |
| truncate1022|subject_z|bvp_eda_resp_small|rf|base | 0.807 | 0.829 | truncate1022 | subject_z | bvp_eda_resp_small | rf |
| truncate1022|subject_z|bvp_eda_resp_small|logreg|base | 0.807 | 0.829 | truncate1022 | subject_z | bvp_eda_resp_small | logreg |
| linear1022|subject_z|bvp_eda_core|xgb|base | 0.807 | 0.829 | linear1022 | subject_z | bvp_eda_core | xgb |
| linear1022|subject_z|bvp_eda_resp_small|logreg|base | 0.807 | 0.829 | linear1022 | subject_z | bvp_eda_resp_small | logreg |

## Best Anchor Refinements

| base | refine | macro-F1 | acc | prec NoPain | rec NoPain |
|---|---|---:|---:|---:|---:|
| truncate1022|subject_robust|bvp_eda_core|xgb|base | anchor_center_l05 | 0.833 | 0.852 | 0.778 | 0.778 |
| truncate1022|subject_robust|bvp_eda_core|xgb|base | anchor_z_l05 | 0.828 | 0.847 | 0.771 | 0.771 |
| poly1022|subject_robust|bvp_eda_core|xgb|base | anchor_z_l05 | 0.828 | 0.847 | 0.771 | 0.771 |
| truncate1022|subject_robust|bvp_eda_core|xgb|base | anchor_z_l10 | 0.823 | 0.843 | 0.764 | 0.764 |
| poly1022|subject_robust|bvp_eda_core|xgb|base | anchor_center_l05 | 0.823 | 0.843 | 0.764 | 0.764 |
| truncate1022|subject_z|bvp_eda_core|xgb|base | anchor_z_l05 | 0.823 | 0.843 | 0.764 | 0.764 |
| truncate1022|subject_z|bvp_eda_core|xgb|base | anchor_z_l10 | 0.823 | 0.843 | 0.764 | 0.764 |
| linear1022|subject_robust|bvp_eda_core|xgb|base | anchor_center_l05 | 0.823 | 0.843 | 0.764 | 0.764 |
| truncate1022|subject_robust|bvp_eda_resp_small|xgb|base | anchor_z_l05 | 0.823 | 0.843 | 0.764 | 0.764 |
| linear1022|subject_robust|bvp_eda_core|xgb|base | anchor_z_l05 | 0.818 | 0.838 | 0.757 | 0.757 |
| linear1022|subject_robust|bvp_eda_core|xgb|base | anchor_z_l10 | 0.818 | 0.838 | 0.757 | 0.757 |
| truncate1022|subject_robust|bvp_eda_resp_small|xgb|base | anchor_center_l05 | 0.818 | 0.838 | 0.757 | 0.757 |