# AI4Pain 2026

Cleaned final workspace for the `NoPain / PainArm / PainHand` challenge.

## Main Scripts

1. `scripts/run_stage1.py`
   Binary `NoPain vs Pain` pipeline.

2. `scripts/run_stage2.py`
   Binary `PainArm vs PainHand` localization pipeline on pain windows.

3. `scripts/run_combined.py`
   End-to-end 3-class two-stage pipeline.

4. `scripts/run_multiclass.py`
   End-to-end 3-class single-classifier pipeline.

5. `scripts/generate_modality_tables.py`
   Recreates paper-style modality comparison tables.

## Default Best Configs

### Stage 1
- `truncate1022`
- `subject_robust`
- `bvp_eda_core`
- `xgb`
- `anchor_center_l05`

### Stage 2
- `truncate1022`
- `subject_z`
- `resp_all`
- `logreg`
- `robust` scaler
- `isotonic` calibration

### Combined
- stage 1: best config above
- stage 2: best config above
- decoder: `joint_weighted`
- weights: `w0=0.8`, `w1=1.2`, `w2=1.4`

## Example Runs

```bash
uv run python scripts/run_stage1.py
uv run python scripts/run_stage2.py
uv run python scripts/run_combined.py
uv run python scripts/run_multiclass.py
uv run python scripts/generate_modality_tables.py
```

## Important Retained Artifacts

- `results/reports/FINAL_FINDINGS.md`
- `results/reports/31_two_stage_subject_adapt_suite.md`
- `results/reports/32_stage1_upgrade_suite.md`
- `results/reports/33_stage1_svm_check.md`
- `plots/suite31/`
- `plots/suite32/`
- `plots/suite33/`

## Feature Tables Kept

- `results/tables/all_features_merged_1022.parquet`
- `results/tables/all_features_merged_linear1022.parquet`
- `results/tables/all_features_merged_poly1022.parquet`
