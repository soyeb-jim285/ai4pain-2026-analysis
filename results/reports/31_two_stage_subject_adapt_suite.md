# 31 — Two-stage subject-adaptation suite

- runtime: 966.1s
- evaluated full pipeline configs: 96 (cap <= 100)
- suite dimensions: 3 input modes x 4 stage-1 variants x 4 stage-2 variants x 2 decoders = 96

## Best Overall

- config_id: truncate1022|xgb_subject_z|resp_all_robust|joint_weighted
- resample_tag: truncate1022
- stage1_id: xgb_subject_z
- stage2_id: resp_all_robust
- decoder_id: joint_weighted
- macro_f1_mean: 0.6064814814814815
- accuracy_mean: 0.6064814814814815

## Top Validation Configs

| config | macro-F1 | acc | resample | stage1 | stage2 | decoder |
|---|---:|---:|---|---|---|---|
| truncate1022|xgb_subject_z|resp_all_robust|joint_weighted | 0.606 | 0.606 | truncate1022 | xgb_subject_z | resp_all_robust | joint_weighted |
| poly1022|xgb_subject_robust|anchor_z_robust|joint_weighted | 0.602 | 0.602 | poly1022 | xgb_subject_robust | anchor_z_robust | joint_weighted |
| poly1022|xgb_subject_robust|resp_all_std|joint_weighted | 0.595 | 0.595 | poly1022 | xgb_subject_robust | resp_all_std | joint_weighted |
| truncate1022|logreg_subject_z|resp_all_std|joint_weighted | 0.590 | 0.590 | truncate1022 | logreg_subject_z | resp_all_std | joint_weighted |
| truncate1022|xgb_subject_z|resp_all_robust|split_topk | 0.588 | 0.588 | truncate1022 | xgb_subject_z | resp_all_robust | split_topk |
| linear1022|xgb_subject_z|anchor_z_robust|joint_weighted | 0.586 | 0.586 | linear1022 | xgb_subject_z | anchor_z_robust | joint_weighted |
| poly1022|xgb_subject_robust|resp_all_robust|joint_weighted | 0.586 | 0.586 | poly1022 | xgb_subject_robust | resp_all_robust | joint_weighted |
| poly1022|xgb_subject_robust|anchor_z_robust|split_topk | 0.586 | 0.586 | poly1022 | xgb_subject_robust | anchor_z_robust | split_topk |
| truncate1022|xgb_subject_z|resp_all_std|joint_weighted | 0.583 | 0.583 | truncate1022 | xgb_subject_z | resp_all_std | joint_weighted |
| truncate1022|xgb_subject_z|anchor_z_robust|joint_weighted | 0.583 | 0.583 | truncate1022 | xgb_subject_z | anchor_z_robust | joint_weighted |
| truncate1022|xgb_subject_robust|resp_all_robust|split_topk | 0.583 | 0.583 | truncate1022 | xgb_subject_robust | resp_all_robust | split_topk |
| truncate1022|xgb_subject_robust|resp_all_std|split_topk | 0.583 | 0.583 | truncate1022 | xgb_subject_robust | resp_all_std | split_topk |

## Best Stage-1 Candidates

| resample | stage1 | macro-F1 raw | macro-F1 cal |
|---|---|---:|---:|
| poly1022 | xgb_subject_robust | 0.828 | 0.828 |
| linear1022 | xgb_subject_z | 0.818 | 0.818 |
| linear1022 | xgb_subject_robust | 0.812 | 0.812 |
| truncate1022 | xgb_subject_z | 0.807 | 0.807 |
| truncate1022 | xgb_subject_robust | 0.807 | 0.807 |
| poly1022 | xgb_subject_z | 0.807 | 0.807 |
| linear1022 | logreg_subject_z | 0.802 | 0.802 |
| truncate1022 | logreg_subject_z | 0.802 | 0.802 |
| poly1022 | logreg_subject_z | 0.792 | 0.792 |
| poly1022 | xgb_global | 0.781 | 0.781 |
| linear1022 | xgb_global | 0.771 | 0.771 |
| truncate1022 | xgb_global | 0.766 | 0.766 |