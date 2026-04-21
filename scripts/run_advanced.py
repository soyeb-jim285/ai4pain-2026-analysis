"""Advanced boosts: multi-view ensemble, feature mixup, subject-residual pretrain.

Tests 6 variants:
1. baseline (truncate only)
2. multiview (avg probs across truncate + linear + poly)
3. mixup (intra-class feature interpolation)
4. subject_residual (features minus subject-mean via linear regression)
5. multiview + mixup
6. multiview + mixup + residual
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.final_pipeline import (
    ARM_HAND, ModelSpec, armhand_binary, build_norm_map, class_codes_3,
    decode_joint_weighted, fit_binary_calibrator, fit_binary_proba,
    load_clean_features, metrics_multiclass, plot_confusion,
    stage1_feature_sets, stage2_feature_sets,
)

RNG = np.random.default_rng(42)


def mixup_features(df, feat_cols, alpha=0.4, n_extra_per_subject=12):
    """Intra-class mixup: within class+subject pair samples linearly."""
    augmented = []
    rng = np.random.default_rng(42)
    for subj in df["subject"].unique():
        for cls in df["class"].unique():
            grp = df[(df["subject"] == subj) & (df["class"] == cls)]
            if len(grp) < 2:
                continue
            vals = grp[feat_cols].to_numpy(dtype=np.float32)
            for _ in range(n_extra_per_subject):
                i, j = rng.integers(0, len(grp), 2)
                lam = rng.beta(alpha, alpha)
                mix = lam * vals[i] + (1 - lam) * vals[j]
                row = grp.iloc[0].copy()
                for k, c in enumerate(feat_cols):
                    row[c] = float(mix[k])
                augmented.append(row)
    if not augmented:
        return df
    aug_df = pd.DataFrame(augmented)
    return pd.concat([df, aug_df], ignore_index=True)


def subject_residual(df, feat_cols):
    """Subtract subject-mean predicted features: residual = x - ridge_fit(subject_onehot -> x)."""
    subjects = sorted(df["subject"].unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    S = np.zeros((len(df), len(subjects)), dtype=np.float32)
    for i, s in enumerate(df["subject"].to_numpy()):
        S[i, subj_to_idx[s]] = 1
    X = df[feat_cols].to_numpy(dtype=np.float32)
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit(S, X)
    X_pred = ridge.predict(S)
    X_res = (X - X_pred).astype(np.float32)
    out = df.copy()
    out[feat_cols] = X_res
    return out


def fit_s1(df_norm, feat_cols, use_mixup=False):
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feats = stage1_feature_sets(train, feat_cols)["bvp_eda_core"]
    spec = ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    for subj in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subj].reset_index(drop=True)
        te = train[train["subject"] == subj].reset_index(drop=True)
        if use_mixup:
            tr = mixup_features(tr, feats, alpha=0.4, n_extra_per_subject=8)
        p = fit_binary_proba(tr, te, feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
        train_probs[train["subject"] == subj] = p
    if use_mixup:
        train_aug = mixup_features(train, feats, alpha=0.4, n_extra_per_subject=8)
    else:
        train_aug = train
    val_probs = fit_binary_proba(train_aug, val, feats, (train_aug["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator("sigmoid", train_probs[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return train, val, 1 - cal(train_probs[:, 0]), 1 - cal(val_probs[:, 0])


def fit_s2(df_norm, feat_cols, use_mixup=False):
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
        if use_mixup:
            pain_tr = mixup_features(pain_tr, feats, alpha=0.4, n_extra_per_subject=8)
        p = fit_binary_proba(pain_tr, te, feats, armhand_binary(pain_tr["class"]), "robust", spec)
        train_probs[train_full["subject"] == subj] = p
    if use_mixup:
        pain_train_aug = mixup_features(pain_train, feats, alpha=0.4, n_extra_per_subject=8)
    else:
        pain_train_aug = pain_train
    val_probs = fit_binary_proba(pain_train_aug, val_full, feats, armhand_binary(pain_train_aug["class"]), "robust", spec)
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


def evaluate_one(df_all, feat_cols, use_mixup=False, use_residual=False):
    if use_residual:
        df_all = subject_residual(df_all, feat_cols)
    norm_map = build_norm_map(df_all, feat_cols)
    train_s1, val_s1, train_pain, val_pain = fit_s1(norm_map["subject_robust"], feat_cols, use_mixup)
    train_s2, val_s2, train_arm, val_arm = fit_s2(norm_map["subject_z"], feat_cols, use_mixup)
    v_arm_al = align(val_s2, val_arm, val_s1)
    t_arm_al = align(train_s2, train_arm, train_s1)
    return train_s1, val_s1, train_pain, val_pain, t_arm_al, v_arm_al


def evaluate_multiview(feat_paths, use_mixup=False, use_residual=False):
    pain_train_list, arm_train_list = [], []
    pain_val_list, arm_val_list = [], []
    ref_train, ref_val = None, None
    for fp in feat_paths:
        df_all, feat_cols = load_clean_features(fp)
        train_s1, val_s1, train_pain, val_pain, t_arm, v_arm = evaluate_one(df_all, feat_cols, use_mixup, use_residual)
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
    pain_train = np.mean(pain_train_list, axis=0)
    pain_val = np.mean(pain_val_list, axis=0)
    arm_train = np.mean(arm_train_list, axis=0)
    arm_val = np.mean(arm_val_list, axis=0)
    return ref_train, ref_val, pain_train, pain_val, arm_train, arm_val


def main() -> None:
    out_dir = Path("results/final/advanced")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_all = [
        "results/tables/all_features_merged_1022.parquet",
        "results/tables/all_features_merged_linear1022.parquet",
        "results/tables/all_features_merged_poly1022.parquet",
    ]
    paths_single = [paths_all[0]]

    variants = [
        ("baseline", paths_single, False, False),
        ("multiview", paths_all, False, False),
        ("mixup", paths_single, True, False),
        ("residual", paths_single, False, True),
        ("mv+mixup", paths_all, True, False),
        ("mv+mixup+resid", paths_all, True, True),
    ]
    results = []
    for name, paths, use_mixup, use_res in variants:
        print(f"\n>>> {name} (mixup={use_mixup} residual={use_res} views={len(paths)})")
        ref_train, ref_val, pain_t, pain_v, arm_t, arm_v = evaluate_multiview(paths, use_mixup, use_res)
        y_true_v = class_codes_3(ref_val["class"])
        y_true_t = class_codes_3(ref_train["class"])
        y_pred_v = decode(ref_val, pain_v, arm_v, 1, 1, 1)
        y_pred_t = decode(ref_train, pain_t, arm_t, 1, 1, 1)
        m_v = metrics_multiclass(y_true_v, y_pred_v)
        m_t = metrics_multiclass(y_true_t, y_pred_t)
        print(f"  LOSO f1={m_t['macro_f1']:.4f} acc={m_t['accuracy']:.4f}  |  VAL f1={m_v['macro_f1']:.4f} acc={m_v['accuracy']:.4f}")
        results.append({
            "variant": name,
            "loso_f1": m_t["macro_f1"], "loso_acc": m_t["accuracy"],
            "val_f1": m_v["macro_f1"], "val_acc": m_v["accuracy"],
        })

    df = pd.DataFrame(results).sort_values("val_f1", ascending=False)
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== RANKING (by val_f1) ===")
    print(df.to_string(index=False))
    df2 = df.sort_values("loso_f1", ascending=False)
    print("\n=== RANKING (by loso_f1) ===")
    print(df2.to_string(index=False))


if __name__ == "__main__":
    main()
