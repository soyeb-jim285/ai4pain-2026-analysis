"""Per-subject LOSO F1 stats: mean, std, min, max."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND, ModelSpec, armhand_binary, build_norm_map, class_codes_3,
    decode_joint_weighted, fit_binary_calibrator, fit_binary_proba,
    load_clean_features, metrics_multiclass,
    stage1_feature_sets, stage2_feature_sets,
)


def fit_s1(df_norm, feat_cols):
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feats = stage1_feature_sets(train, feat_cols)["bvp_eda_core"]
    spec = ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    for subj in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subj].reset_index(drop=True)
        te = train[train["subject"] == subj].reset_index(drop=True)
        p = fit_binary_proba(tr, te, feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
        train_probs[train["subject"] == subj] = p
    val_probs = fit_binary_proba(train, val, feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator("sigmoid", train_probs[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return train, val, 1 - cal(train_probs[:, 0]), 1 - cal(val_probs[:, 0])


def fit_s2(df_norm, feat_cols):
    train_full = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val_full = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feats = stage2_feature_sets(train_full, feat_cols)["resp_all"]
    spec = ModelSpec("logreg", {"C": 1.0})
    pain_train = train_full[train_full["class"].isin(ARM_HAND)].reset_index(drop=True)
    train_probs = np.zeros((len(train_full), 2), dtype=np.float32)
    for subj in sorted(train_full["subject"].unique()):
        pain_tr = pain_train[pain_train["subject"] != subj].reset_index(drop=True)
        te = train_full[train_full["subject"] == subj].reset_index(drop=True)
        if len(pain_tr) == 0:
            continue
        p = fit_binary_proba(pain_tr, te, feats, armhand_binary(pain_tr["class"]), "robust", spec)
        train_probs[train_full["subject"] == subj] = p
    val_probs = fit_binary_proba(pain_train, val_full, feats, armhand_binary(pain_train["class"]), "robust", spec)
    pain_mask = (train_full["class"].isin(ARM_HAND)).to_numpy()
    y_arm = (train_full.loc[pain_mask, "class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator("isotonic", train_probs[pain_mask, 0], y_arm)
    return train_full, val_full, cal(train_probs[:, 0]).astype(np.float32), cal(val_probs[:, 0]).astype(np.float32)


def decode(df, pain_p, arm_p, w0=1, w1=1, w2=1):
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


def run_one(paths):
    pain_train_list, arm_train_list = [], []
    pain_val_list, arm_val_list = [], []
    ref_train, ref_val = None, None
    for fp in paths:
        df_all, feat_cols = load_clean_features(fp)
        norm_map = build_norm_map(df_all, feat_cols)
        train_s1, val_s1, train_pain, val_pain = fit_s1(norm_map["subject_robust"], feat_cols)
        train_s2, val_s2, train_arm, val_arm = fit_s2(norm_map["subject_z"], feat_cols)
        v_arm = align(val_s2, val_arm, val_s1)
        t_arm = align(train_s2, train_arm, train_s1)
        if ref_train is None:
            ref_train, ref_val = train_s1, val_s1
            pain_train_list.append(train_pain)
            arm_train_list.append(t_arm)
            pain_val_list.append(val_pain)
            arm_val_list.append(v_arm)
        else:
            pain_train_list.append(align(train_s1, train_pain, ref_train))
            arm_train_list.append(align(train_s1, t_arm, ref_train))
            pain_val_list.append(align(val_s1, val_pain, ref_val))
            arm_val_list.append(align(val_s1, v_arm, ref_val))
    return (ref_train, ref_val,
            np.mean(pain_train_list, axis=0), np.mean(pain_val_list, axis=0),
            np.mean(arm_train_list, axis=0), np.mean(arm_val_list, axis=0))


def per_subject(df, pain_p, arm_p):
    y_true = class_codes_3(df["class"])
    y_pred = decode(df, pain_p, arm_p, 1, 1, 1)
    rows = []
    for subj in sorted(df["subject"].unique()):
        mask = (df["subject"] == subj).to_numpy()
        m = metrics_multiclass(y_true[mask], y_pred[mask])
        rows.append({"subject": int(subj), "f1": m["macro_f1"], "acc": m["accuracy"]})
    return pd.DataFrame(rows)


def summarize(df, label):
    s = df["f1"].describe()
    return {
        "config": label,
        "mean": s["mean"], "std": s["std"],
        "min": s["min"], "max": s["max"],
        "median": s["50%"], "q25": s["25%"], "q75": s["75%"],
        "n": int(s["count"]),
    }


def main() -> None:
    out_dir = Path("results/final/loso_spread")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_all = [
        "results/tables/all_features_merged_1022.parquet",
        "results/tables/all_features_merged_linear1022.parquet",
        "results/tables/all_features_merged_poly1022.parquet",
    ]
    configs = [
        ("baseline", [paths_all[0]]),
        ("multiview", paths_all),
    ]
    summaries = []
    for name, paths in configs:
        print(f"\n>>> {name}")
        ref_t, ref_v, pain_t, pain_v, arm_t, arm_v = run_one(paths)
        ps_loso = per_subject(ref_t, pain_t, arm_t)
        ps_val = per_subject(ref_v, pain_v, arm_v)
        ps_loso.to_csv(out_dir / f"{name}_loso_per_subject.csv", index=False)
        ps_val.to_csv(out_dir / f"{name}_val_per_subject.csv", index=False)
        summaries.append(summarize(ps_loso, f"{name} LOSO"))
        summaries.append(summarize(ps_val, f"{name} VAL"))
        print(f"  LOSO per-subject:")
        print(ps_loso.sort_values("f1").to_string(index=False))
        print(f"  VAL per-subject:")
        print(ps_val.sort_values("f1").to_string(index=False))

    df = pd.DataFrame(summaries)
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
