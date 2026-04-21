"""Boosted pipeline across 3 resample modes (truncate, linear, poly) + augmentation."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND, ModelSpec, armhand_binary, build_norm_map, class_codes_3,
    decode_joint_weighted, fit_binary_calibrator, fit_binary_proba,
    load_clean_features, metrics_multiclass, plot_confusion,
    stage1_feature_sets, stage2_feature_sets,
)


def fit_s1(df_norm, feat_cols, jitter_sigma=0.0):
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feats = stage1_feature_sets(train, feat_cols)["bvp_eda_core"]
    spec = ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    rng = np.random.default_rng(42)
    for subj in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subj].reset_index(drop=True)
        te = train[train["subject"] == subj].reset_index(drop=True)
        if jitter_sigma > 0:
            tr_aug = tr.copy()
            for c in feats:
                tr_aug[c] = tr_aug[c] + rng.normal(0, jitter_sigma, len(tr))
            tr = pd.concat([tr, tr_aug], ignore_index=True)
        p = fit_binary_proba(tr, te, feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
        train_probs[train["subject"] == subj] = p
    val_probs = fit_binary_proba(train, val, feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator("sigmoid", train_probs[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return train, val, 1 - cal(train_probs[:, 0]), 1 - cal(val_probs[:, 0])


def fit_s2(df_norm, feat_cols, jitter_sigma=0.0):
    train_full = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val_full = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feats = stage2_feature_sets(train_full, feat_cols)["resp_all"]
    spec = ModelSpec("logreg", {"C": 1.0})
    pain_train = train_full[train_full["class"].isin(ARM_HAND)].reset_index(drop=True)
    train_probs = np.zeros((len(train_full), 2), dtype=np.float32)
    rng = np.random.default_rng(43)
    for subj in sorted(train_full["subject"].unique()):
        pain_tr = pain_train[pain_train["subject"] != subj].reset_index(drop=True)
        te = train_full[train_full["subject"] == subj].reset_index(drop=True)
        if len(pain_tr) == 0:
            continue
        if jitter_sigma > 0:
            pt_aug = pain_tr.copy()
            for c in feats:
                pt_aug[c] = pt_aug[c] + rng.normal(0, jitter_sigma, len(pain_tr))
            pain_tr = pd.concat([pain_tr, pt_aug], ignore_index=True)
        p = fit_binary_proba(pain_tr, te, feats, armhand_binary(pain_tr["class"]), "robust", spec)
        train_probs[train_full["subject"] == subj] = p
    val_probs = fit_binary_proba(pain_train, val_full, feats, armhand_binary(pain_train["class"]), "robust", spec)
    pain_mask = (train_full["class"].isin(ARM_HAND)).to_numpy()
    y_arm = (train_full.loc[pain_mask, "class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator("isotonic", train_probs[pain_mask, 0], y_arm)
    return train_full, val_full, cal(train_probs[:, 0]).astype(np.float32), cal(val_probs[:, 0]).astype(np.float32)


def decode(df, pain_p, arm_p, w0, w1, w2):
    s1 = np.column_stack([1 - pain_p, pain_p]).astype(np.float32)
    s2 = np.column_stack([arm_p, 1 - arm_p]).astype(np.float32)
    y_pred = np.zeros(len(df), dtype=int)
    for subj in sorted(df["subject"].unique()):
        mask = (df["subject"] == subj).to_numpy()
        y_pred[mask] = decode_joint_weighted(s1[mask], s2[mask], w0=w0, w1=w1, w2=w2)
    return y_pred


def align(src_df, src_vals, ref_df):
    mp = dict(zip(src_df["segment_id"], src_vals))
    return np.array([mp[sid] for sid in ref_df["segment_id"]], dtype=np.float32)


def evaluate(df_all, feat_cols, jitter_sigma=0.0, tag=""):
    norm_map = build_norm_map(df_all, feat_cols)
    train_s1, val_s1, train_pain, val_pain = fit_s1(norm_map["subject_robust"], feat_cols, jitter_sigma)
    train_s2, val_s2, train_arm, val_arm = fit_s2(norm_map["subject_z"], feat_cols, jitter_sigma)
    v_arm_al = align(val_s2, val_arm, val_s1)
    t_arm_al = align(train_s2, train_arm, train_s1)
    y_true_val = class_codes_3(val_s1["class"])
    y_true_trn = class_codes_3(train_s1["class"])
    y_pred_val = decode(val_s1, val_pain, v_arm_al, 1, 1, 1)
    y_pred_trn = decode(train_s1, train_pain, t_arm_al, 1, 1, 1)
    return {
        "tag": tag,
        "loso_f1": metrics_multiclass(y_true_trn, y_pred_trn)["macro_f1"],
        "val_f1": metrics_multiclass(y_true_val, y_pred_val)["macro_f1"],
    }


def main() -> None:
    out_dir = Path("results/final/boosted_resample")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("truncate1022", "results/tables/all_features_merged_1022.parquet", 0.0),
        ("truncate1022_jit05", "results/tables/all_features_merged_1022.parquet", 0.05),
        ("truncate1022_jit10", "results/tables/all_features_merged_1022.parquet", 0.10),
        ("linear1022", "results/tables/all_features_merged_linear1022.parquet", 0.0),
        ("linear1022_jit05", "results/tables/all_features_merged_linear1022.parquet", 0.05),
        ("poly1022", "results/tables/all_features_merged_poly1022.parquet", 0.0),
        ("poly1022_jit05", "results/tables/all_features_merged_poly1022.parquet", 0.05),
    ]
    results = []
    for tag, fp, jitter in configs:
        if not Path(fp).exists():
            print(f"SKIP {tag}: missing {fp}")
            continue
        print(f"\n>>> {tag} (jitter={jitter})")
        df_all, feat_cols = load_clean_features(fp)
        r = evaluate(df_all, feat_cols, jitter_sigma=jitter, tag=tag)
        print(f"  LOSO f1={r['loso_f1']:.4f}  VAL f1={r['val_f1']:.4f}")
        results.append(r)

    df = pd.DataFrame(results).sort_values("val_f1", ascending=False)
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== RANKING ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
