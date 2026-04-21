# Final Findings

## Best Stage 1

- config: `truncate1022 | subject_robust | bvp_eda_core | xgb | anchor_center_l05`
- validation macro-F1: `0.8333`
- validation accuracy: `0.8519`

Main takeaways:
- `BVP + EDA` is clearly the best stage-1 signal pool.
- `subject_robust` beats `subject_z` on average.
- a light NoPain-anchor refinement helps.

## Best Stage 2

- config family: `RESP`-first arm-vs-hand model
- strongest stable recipe: `subject_z | resp_all | logreg | robust scaler | isotonic calibration`

Main takeaways:
- `RESP` carries the strongest arm-vs-hand localization signal.
- larger model families did not beat the simple calibrated linear model reliably.

## Best Combined 3-Class Pipeline

- config: `truncate1022 | xgb_subject_z | resp_all_robust | joint_weighted`
- validation macro-F1: `0.6065`

Main takeaways:
- the best overall gain came from `calibration + weighted exact-count decoding`.
- `1022` was better than `1000`.
- explicit resampling (`linear1022`, `poly1022`) did not beat `truncate1022` overall.

## Key Experiments That Helped

1. `1022` windows instead of `1000`
2. exact `12/12/12` subject-level decoding
3. calibrated and weighted joint decoding
4. upgraded stage-1 binary detector using `BVP + EDA`

## Key Experiments That Did Not Win

1. latent intensity split
2. pairwise ranking stage 2
3. heavy stage-2 reformulations
4. linear resampling
5. replacing XGBoost stage 1 with SVM

## SVM Check

- RBF SVM was better than linear SVM.
- best SVM stage-1 validation macro-F1: `0.8177`
- still below the XGBoost stage-1 winner by about `0.0156`

## Files To Trust Most

- `results/reports/31_two_stage_subject_adapt_suite.md`
- `results/reports/32_stage1_upgrade_suite.md`
- `results/reports/33_stage1_svm_check.md`
