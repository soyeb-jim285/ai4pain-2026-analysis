# Stage 2 Optimization Sweep

- configs: 40

## Top 10 by LOSO acc

| model   | feature_set    | norm           | calibration   | anchor   | scaler   |   n_features |   loso_acc_mean |   loso_acc_std |   loso_acc_min |   val_acc_mean |   val_acc_std |   val_acc_min |   loso_auc |   val_auc |
|:--------|:---------------|:---------------|:--------------|:---------|:---------|-------------:|----------------:|---------------:|---------------:|---------------:|--------------:|--------------:|-----------:|----------:|
| logreg  | all_top40      | subject_robust | isotonic      | none     | robust   |           40 |          0.5732 |         0.1060 |         0.3333 |         0.5347 |        0.0865 |        0.4167 |     0.5900 |    0.5488 |
| logreg  | bvp_resp_top30 | subject_z      | isotonic      | none     | robust   |           30 |          0.5732 |         0.0977 |         0.3333 |         0.5486 |        0.0865 |        0.4167 |     0.5906 |    0.5662 |
| logreg  | eda_resp_top30 | subject_robust | isotonic      | none     | robust   |           30 |          0.5630 |         0.1070 |         0.3333 |         0.5556 |        0.0982 |        0.4167 |     0.5852 |    0.5369 |
| logreg  | eda_resp_top30 | subject_robust | sigmoid       | none     | robust   |           30 |          0.5589 |         0.1045 |         0.3333 |         0.5486 |        0.0633 |        0.4167 |     0.5675 |    0.5457 |
| logreg  | resp_top20     | subject_robust | isotonic      | none     | robust   |           20 |          0.5589 |         0.0995 |         0.3333 |         0.5208 |        0.0908 |        0.3333 |     0.5702 |    0.5371 |
| logreg  | resp_top20     | subject_z      | isotonic      | none     | robust   |           20 |          0.5569 |         0.0982 |         0.4167 |         0.5486 |        0.0533 |        0.5000 |     0.5783 |    0.5595 |
| logreg  | bvp_resp_top30 | subject_z      | sigmoid       | none     | robust   |           30 |          0.5549 |         0.0984 |         0.3333 |         0.5278 |        0.0856 |        0.3333 |     0.5755 |    0.5598 |
| logreg  | resp_top20     | subject_robust | sigmoid       | none     | robust   |           20 |          0.5528 |         0.0987 |         0.3333 |         0.5208 |        0.0770 |        0.4167 |     0.5536 |    0.5392 |
| logreg  | all_top40      | subject_z      | isotonic      | none     | robust   |           40 |          0.5528 |         0.1069 |         0.3333 |         0.5486 |        0.0865 |        0.4167 |     0.5821 |    0.5700 |
| logreg  | eda_resp_top30 | subject_z      | isotonic      | none     | robust   |           30 |          0.5528 |         0.0987 |         0.3333 |         0.5278 |        0.0708 |        0.4167 |     0.5694 |    0.5467 |

## All (sorted)

| model   | feature_set    | norm           | calibration   | anchor   | scaler   |   n_features |   loso_acc_mean |   loso_acc_std |   loso_acc_min |   val_acc_mean |   val_acc_std |   val_acc_min |   loso_auc |   val_auc |
|:--------|:---------------|:---------------|:--------------|:---------|:---------|-------------:|----------------:|---------------:|---------------:|---------------:|--------------:|--------------:|-----------:|----------:|
| logreg  | all_top40      | subject_robust | isotonic      | none     | robust   |           40 |          0.5732 |         0.1060 |         0.3333 |         0.5347 |        0.0865 |        0.4167 |     0.5900 |    0.5488 |
| logreg  | bvp_resp_top30 | subject_z      | isotonic      | none     | robust   |           30 |          0.5732 |         0.0977 |         0.3333 |         0.5486 |        0.0865 |        0.4167 |     0.5906 |    0.5662 |
| logreg  | eda_resp_top30 | subject_robust | isotonic      | none     | robust   |           30 |          0.5630 |         0.1070 |         0.3333 |         0.5556 |        0.0982 |        0.4167 |     0.5852 |    0.5369 |
| logreg  | eda_resp_top30 | subject_robust | sigmoid       | none     | robust   |           30 |          0.5589 |         0.1045 |         0.3333 |         0.5486 |        0.0633 |        0.4167 |     0.5675 |    0.5457 |
| logreg  | resp_top20     | subject_robust | isotonic      | none     | robust   |           20 |          0.5589 |         0.0995 |         0.3333 |         0.5208 |        0.0908 |        0.3333 |     0.5702 |    0.5371 |
| logreg  | resp_top20     | subject_z      | isotonic      | none     | robust   |           20 |          0.5569 |         0.0982 |         0.4167 |         0.5486 |        0.0533 |        0.5000 |     0.5783 |    0.5595 |
| logreg  | bvp_resp_top30 | subject_z      | sigmoid       | none     | robust   |           30 |          0.5549 |         0.0984 |         0.3333 |         0.5278 |        0.0856 |        0.3333 |     0.5755 |    0.5598 |
| logreg  | resp_top20     | subject_robust | sigmoid       | none     | robust   |           20 |          0.5528 |         0.0987 |         0.3333 |         0.5208 |        0.0770 |        0.4167 |     0.5536 |    0.5392 |
| logreg  | all_top40      | subject_z      | isotonic      | none     | robust   |           40 |          0.5528 |         0.1069 |         0.3333 |         0.5486 |        0.0865 |        0.4167 |     0.5821 |    0.5700 |
| logreg  | eda_resp_top30 | subject_z      | isotonic      | none     | robust   |           30 |          0.5528 |         0.0987 |         0.3333 |         0.5278 |        0.0708 |        0.4167 |     0.5694 |    0.5467 |
| logreg  | all_top40      | subject_robust | sigmoid       | none     | robust   |           40 |          0.5528 |         0.1189 |         0.3333 |         0.5139 |        0.0666 |        0.4167 |     0.5730 |    0.5488 |
| logreg  | resp_all       | subject_robust | isotonic      | none     | robust   |           44 |          0.5508 |         0.1006 |         0.2500 |         0.5486 |        0.0989 |        0.4167 |     0.5591 |    0.5658 |
| logreg  | resp_all       | subject_robust | sigmoid       | none     | robust   |           44 |          0.5488 |         0.0991 |         0.3333 |         0.5556 |        0.0856 |        0.4167 |     0.5447 |    0.5694 |
| logreg  | resp_top20     | subject_z      | sigmoid       | none     | robust   |           20 |          0.5467 |         0.1058 |         0.3333 |         0.5417 |        0.0538 |        0.4167 |     0.5623 |    0.5647 |
| logreg  | bvp_resp_top30 | subject_robust | isotonic      | none     | robust   |           30 |          0.5447 |         0.0922 |         0.3333 |         0.5278 |        0.0708 |        0.4167 |     0.5811 |    0.5466 |
| logreg  | all_top40      | subject_z      | sigmoid       | none     | robust   |           40 |          0.5427 |         0.1059 |         0.3333 |         0.5417 |        0.1049 |        0.4167 |     0.5659 |    0.5601 |
| logreg  | bvp_resp_top30 | subject_robust | sigmoid       | none     | robust   |           30 |          0.5386 |         0.1090 |         0.2500 |         0.5278 |        0.1094 |        0.2500 |     0.5658 |    0.5468 |
| logreg  | eda_resp_top30 | subject_z      | sigmoid       | none     | robust   |           30 |          0.5386 |         0.1043 |         0.3333 |         0.5417 |        0.0538 |        0.4167 |     0.5515 |    0.5585 |
| xgb     | resp_top20     | subject_z      | sigmoid       | none     | robust   |           20 |          0.5366 |         0.1058 |         0.2500 |         0.4653 |        0.0795 |        0.3333 |     0.5337 |    0.4680 |
| xgb     | all_top40      | subject_z      | sigmoid       | none     | robust   |           40 |          0.5346 |         0.1041 |         0.3333 |         0.5208 |        0.0970 |        0.4167 |     0.5284 |    0.5143 |
| xgb     | resp_top20     | subject_z      | isotonic      | none     | robust   |           20 |          0.5305 |         0.0952 |         0.2500 |         0.4653 |        0.0718 |        0.3333 |     0.5547 |    0.4724 |
| xgb     | eda_resp_top30 | subject_robust | isotonic      | none     | robust   |           30 |          0.5305 |         0.1160 |         0.3333 |         0.5069 |        0.1100 |        0.3333 |     0.5405 |    0.5306 |
| xgb     | all_top40      | subject_z      | isotonic      | none     | robust   |           40 |          0.5285 |         0.1035 |         0.3333 |         0.5208 |        0.1279 |        0.3333 |     0.5451 |    0.5024 |
| xgb     | resp_all       | subject_z      | isotonic      | none     | robust   |           44 |          0.5264 |         0.0964 |         0.3333 |         0.4583 |        0.1154 |        0.2500 |     0.5482 |    0.4538 |
| xgb     | resp_all       | subject_robust | sigmoid       | none     | robust   |           44 |          0.5264 |         0.1141 |         0.2500 |         0.5069 |        0.1100 |        0.3333 |     0.5278 |    0.5048 |
| logreg  | resp_all       | subject_z      | isotonic      | none     | robust   |           44 |          0.5224 |         0.0974 |         0.3333 |         0.5139 |        0.0889 |        0.3333 |     0.5540 |    0.5661 |
| xgb     | bvp_resp_top30 | subject_z      | isotonic      | none     | robust   |           30 |          0.5224 |         0.1207 |         0.2500 |         0.5278 |        0.1375 |        0.2500 |     0.5152 |    0.5035 |
| xgb     | resp_all       | subject_z      | sigmoid       | none     | robust   |           44 |          0.5203 |         0.1054 |         0.3333 |         0.4583 |        0.1049 |        0.2500 |     0.5309 |    0.4619 |
| xgb     | resp_top20     | subject_robust | isotonic      | none     | robust   |           20 |          0.5203 |         0.1146 |         0.3333 |         0.5208 |        0.0601 |        0.4167 |     0.5376 |    0.5560 |
| xgb     | bvp_resp_top30 | subject_robust | isotonic      | none     | robust   |           30 |          0.5163 |         0.1028 |         0.3333 |         0.5208 |        0.1135 |        0.4167 |     0.5282 |    0.5152 |
| xgb     | resp_top20     | subject_robust | sigmoid       | none     | robust   |           20 |          0.5163 |         0.1153 |         0.3333 |         0.5417 |        0.1049 |        0.4167 |     0.5175 |    0.5588 |
| logreg  | resp_all       | subject_z      | sigmoid       | none     | robust   |           44 |          0.5142 |         0.0881 |         0.3333 |         0.5833 |        0.0589 |        0.5000 |     0.5322 |    0.5716 |
| xgb     | eda_resp_top30 | subject_z      | isotonic      | none     | robust   |           30 |          0.5142 |         0.1056 |         0.2500 |         0.5000 |        0.0833 |        0.3333 |     0.5404 |    0.4958 |
| xgb     | all_top40      | subject_robust | sigmoid       | none     | robust   |           40 |          0.5142 |         0.0881 |         0.3333 |         0.4861 |        0.1170 |        0.2500 |     0.5095 |    0.4574 |
| xgb     | eda_resp_top30 | subject_robust | sigmoid       | none     | robust   |           30 |          0.5081 |         0.0917 |         0.3333 |         0.5625 |        0.0908 |        0.4167 |     0.5190 |    0.5516 |
| xgb     | eda_resp_top30 | subject_z      | sigmoid       | none     | robust   |           30 |          0.5081 |         0.0988 |         0.3333 |         0.5000 |        0.0680 |        0.3333 |     0.5265 |    0.5018 |
| xgb     | bvp_resp_top30 | subject_robust | sigmoid       | none     | robust   |           30 |          0.5061 |         0.0998 |         0.2500 |         0.5139 |        0.0889 |        0.4167 |     0.5123 |    0.5173 |
| xgb     | bvp_resp_top30 | subject_z      | sigmoid       | none     | robust   |           30 |          0.5000 |         0.0991 |         0.3333 |         0.4444 |        0.0856 |        0.2500 |     0.5069 |    0.4651 |
| xgb     | resp_all       | subject_robust | isotonic      | none     | robust   |           44 |          0.4980 |         0.1081 |         0.2500 |         0.5347 |        0.1046 |        0.3333 |     0.5021 |    0.5107 |
| xgb     | all_top40      | subject_robust | isotonic      | none     | robust   |           40 |          0.4837 |         0.1108 |         0.2500 |         0.5556 |        0.0982 |        0.3333 |     0.5138 |    0.5460 |