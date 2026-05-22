"""Multi-seed K-fold subject CV comparison across 3 candidate configs.

Pools train+val subjects, randomly splits into K subject folds with N seeds.
For each (config, seed, fold): reassigns split column, runs combined pipeline,
captures held-out fold metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_pipeline import (
    load_clean_features, build_norm_map, stage1_feature_sets, stage2_feature_sets,
    class_codes_3, metrics_multiclass, stage_probs_to_3class,
)
from scripts.run_combined import (
    build_stage1_spec, build_stage2_spec, fit_stage1, fit_stage2,
    align_probs_to_ref, decode_subjectwise,
)


CONFIGS = {
    "baseline": dict(
        parquet="results/tables/all_features_merged_1022_edadtlp1.parquet",
        s1="bvp_eda_core", s2="eda_resp_top30",
    ),
    "spo2flag_binary": dict(
        parquet="results/tables/all_features_merged_1022_edadtlp1_spo2flag.parquet",
        s1="bvp_eda_core_spo2flag_binary", s2="eda_resp_top30",
    ),
    "median_lp1_small": dict(
        parquet="results/tables/all_features_merged_1022_edamdlp1.parquet",
        s1="bvp_eda_resp_small", s2="bvp_resp_top30",
    ),
    # Top 5 historical val configs (re-tested under multi-seed CV)
    "top_best_mission_default": dict(
        parquet="results/tables/all_features_merged_1022_edadtlp1.parquet",
        s1="all_top100", s2="bvp_resp_top30",
    ),
    "top_combined_oldbest_xgb": dict(
        parquet="results/tables/all_features_merged_1022.parquet",
        s1="bvp_eda_core", s2="resp_all",
        overrides=dict(stage1_norm="subject_robust", stage1_model="xgb",
                       stage1_calibration="sigmoid", stage1_anchor_mode="center",
                       decoder="joint_weighted"),
    ),
    "top_truncate_xgb_noanchor": dict(
        parquet="results/tables/all_features_merged_1022.parquet",
        s1="bvp_eda_core", s2="resp_all",
        overrides=dict(stage1_norm="subject_robust", stage1_model="xgb",
                       stage1_calibration="sigmoid", stage1_anchor_mode="none",
                       decoder="joint_weighted"),
    ),
    "top_tuning_svm_poly": dict(
        parquet="results/tables/all_features_merged_1022.parquet",
        s1="bvp_eda_resp_small", s2="bvp_resp_top30",
        overrides=dict(stage1_model="svm_poly", stage1_svm_c=1.0),
    ),
    "top_edadtlp1_core_bvpresp": dict(
        parquet="results/tables/all_features_merged_1022_edadtlp1.parquet",
        s1="bvp_eda_core", s2="bvp_resp_top30",
    ),
    "median_lp1_small_spo2flag_binary": dict(
        parquet="results/tables/all_features_merged_1022_edamdlp1_spo2flag.parquet",
        s1="bvp_eda_resp_small_spo2flag_binary", s2="bvp_resp_top30",
    ),
    "median_lp1_small_spo2flag_all": dict(
        parquet="results/tables/all_features_merged_1022_edamdlp1_spo2flag.parquet",
        s1="bvp_eda_resp_small_spo2flag", s2="bvp_resp_top30",
    ),
    "bvpnab_edamdlp1_small_bvpresp": dict(
        parquet="results/tables/all_features_merged_1022_bvpnabedamdlp1.parquet",
        s1="bvp_eda_resp_small", s2="bvp_resp_top30",
    ),
    "bvpnab_edamdlp1_small_edaresp": dict(
        parquet="results/tables/all_features_merged_1022_bvpnabedamdlp1.parquet",
        s1="bvp_eda_resp_small", s2="eda_resp_top30",
    ),
}


def build_args(s1, s2, overrides=None):
    args = argparse.Namespace(
        stage1_norm="subject_z", stage1_feature_set=s1, stage1_model="logreg",
        stage1_scaler="std", stage1_calibration="isotonic",
        stage1_anchor_mode="none", stage1_anchor_lambda=0.5,
        stage1_xgb_n_estimators=200, stage1_xgb_max_depth=4, stage1_xgb_learning_rate=0.08,
        stage1_xgb_subsample=1.0, stage1_xgb_colsample_bytree=1.0, stage1_xgb_min_child_weight=1.0,
        stage1_xgb_reg_alpha=0.0, stage1_xgb_reg_lambda=1.0,
        stage1_rf_n_estimators=400, stage1_rf_max_depth=None,
        stage1_logreg_c=1.0, stage1_logreg_penalty="l2", stage1_logreg_l1_ratio=0.5,
        stage1_svm_c=3.0, stage1_svm_gamma="scale", stage1_svm_degree=2, stage1_svm_coef0=1.0,
        stage2_norm="subject_z", stage2_feature_set=s2, stage2_model="logreg",
        stage2_scaler="robust", stage2_calibration="isotonic", stage2_anchor_mode="none",
        stage2_xgb_n_estimators=200, stage2_xgb_max_depth=4, stage2_xgb_learning_rate=0.08,
        stage2_xgb_subsample=1.0, stage2_xgb_colsample_bytree=1.0, stage2_xgb_min_child_weight=1.0,
        stage2_xgb_reg_alpha=0.0, stage2_xgb_reg_lambda=1.0,
        stage2_rf_n_estimators=400, stage2_rf_max_depth=None,
        stage2_logreg_c=1.0, stage2_logreg_penalty="l2", stage2_logreg_l1_ratio=0.5,
        stage2_svm_c=3.0, stage2_svm_gamma="scale", stage2_svm_degree=2, stage2_svm_coef0=1.0,
        decoder="split_topk", w0=1.0, w1=1.0, w2=1.0,
    )
    if overrides:
        for k, v in overrides.items():
            setattr(args, k, v)
    return args


def run_once(df_all, feat_cols, train_subjects, val_subjects, s1_name, s2_name, overrides=None):
    df = df_all.copy()
    df["split"] = np.where(df["subject"].isin(val_subjects), "validation", "train")
    args = build_args(s1_name, s2_name, overrides)
    norm_map = build_norm_map(df, feat_cols)
    s1n = norm_map[args.stage1_norm]
    train_for_fs = s1n[s1n["split"] == "train"].reset_index(drop=True)
    s1_features = stage1_feature_sets(train_for_fs, feat_cols)[s1_name]
    s1_spec = build_stage1_spec(args)
    train_s1, val_s1, p_tr1, p_va1 = fit_stage1(
        s1n, s1_features, s1_spec, scaler_name=args.stage1_scaler,
        calibration=args.stage1_calibration, anchor_mode=args.stage1_anchor_mode,
        anchor_lambda=args.stage1_anchor_lambda, df_global=df,
    )
    s2n = norm_map[args.stage2_norm]
    train2_for_fs = s2n[s2n["split"] == "train"].reset_index(drop=True)
    s2_features = stage2_feature_sets(train2_for_fs, feat_cols)[s2_name]
    s2_spec = build_stage2_spec(args)
    train_s2, val_s2, p_tr2, p_va2 = fit_stage2(
        s2n, df, s2_features, s2_spec, scaler_name=args.stage2_scaler,
        calibration=args.stage2_calibration, anchor_mode=args.stage2_anchor_mode,
        train_stage1_scores=p_tr1[:, 0], val_stage1_scores=p_va1[:, 0],
    )
    p_tr2 = align_probs_to_ref(train_s1, train_s2, p_tr2)
    p_va2 = align_probs_to_ref(val_s1, val_s2, p_va2)
    y_pred = decode_subjectwise(val_s1, p_va1, p_va2, decoder=args.decoder,
                                w0=args.w0, w1=args.w1, w2=args.w2)
    y_true = class_codes_3(val_s1["class"])
    y_proba = stage_probs_to_3class(p_va1, p_va2)
    return metrics_multiclass(y_true, y_pred, y_proba)


def main(args):
    rng_seeds = list(range(args.seeds))
    n_folds = args.folds
    parquets = {}
    for name, cfg in CONFIGS.items():
        if cfg["parquet"] not in parquets:
            parquets[cfg["parquet"]] = load_clean_features(cfg["parquet"])
    rows = []
    for name, cfg in CONFIGS.items():
        df_all, feat_cols = parquets[cfg["parquet"]]
        subjects = np.array(sorted(df_all["subject"].unique()))
        print(f"\n[{name}] parquet={Path(cfg['parquet']).name} s1={cfg['s1']} s2={cfg['s2']}", flush=True)
        for seed in rng_seeds:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold, (tr_idx, va_idx) in enumerate(kf.split(subjects)):
                tr_subj = subjects[tr_idx].tolist()
                va_subj = subjects[va_idx].tolist()
                m = run_once(df_all, feat_cols, tr_subj, va_subj, cfg["s1"], cfg["s2"], cfg.get("overrides"))
                row = {"config": name, "seed": seed, "fold": fold,
                       "n_val_subjects": len(va_subj), **m}
                rows.append(row)
                print(f"  seed={seed} fold={fold} n_val={len(va_subj):2d}  "
                      f"acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}  auc={m['macro_auc']:.4f}",
                      flush=True)
    df = pd.DataFrame(rows)
    out_fp = Path(args.output)
    df.to_csv(out_fp, index=False)
    print(f"\n[wrote] {out_fp}")
    print("\n=== Aggregate (mean +- std across all seeds*folds) ===")
    agg = df.groupby("config").agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        auc_mean=("macro_auc", "mean"), auc_std=("macro_auc", "std"),
        n=("accuracy", "count"),
    ).round(4)
    print(agg.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--output", default="results/final/cv_compare/results.csv")
    a = p.parse_args()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    main(a)
