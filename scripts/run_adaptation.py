"""Subject adaptation: CORAL + test-time NoPain-anchor recalibration."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.linalg

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, apply_subject_z, armhand_binary,
    channel_of, class_codes_3, decode_joint_weighted, fit_binary_calibrator,
    fit_binary_proba, load_clean_features, metrics_multiclass,
    unique_by_canonical,
)


def coral_transform(Xs, Xt):
    """Align source (train) covariance to target (test subject) covariance."""
    Xs = Xs - Xs.mean(0, keepdims=True)
    Xt_mean = Xt.mean(0, keepdims=True)
    Xt = Xt - Xt_mean
    Cs = np.cov(Xs, rowvar=False) + 1e-3 * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + 1e-3 * np.eye(Xt.shape[1])
    A = scipy.linalg.sqrtm(scipy.linalg.inv(Cs)) @ scipy.linalg.sqrtm(Ct)
    A = np.real(A)
    Xs_aligned = Xs @ A
    return Xs_aligned, A


def fit_s1_coral(df_all, feat_cols, s1_feats, use_coral=False):
    df_norm = apply_subject_robust(df_all, feat_cols)
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    spec = ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    for subj in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subj].reset_index(drop=True)
        te = train[train["subject"] == subj].reset_index(drop=True)
        if use_coral:
            Xs = tr[s1_feats].to_numpy(dtype=np.float32)
            Xt = te[s1_feats].to_numpy(dtype=np.float32)
            Xs_a, _ = coral_transform(Xs, Xt)
            tr = tr.copy()
            tr[s1_feats] = Xs_a.astype(np.float32)
        p = fit_binary_proba(tr, te, s1_feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
        train_probs[train["subject"] == subj] = p
    if use_coral:
        Xs = train[s1_feats].to_numpy(dtype=np.float32)
        Xt = val[s1_feats].to_numpy(dtype=np.float32)
        Xs_a, _ = coral_transform(Xs, Xt)
        train_c = train.copy()
        train_c[s1_feats] = Xs_a.astype(np.float32)
    else:
        train_c = train
    val_probs = fit_binary_proba(train_c, val, s1_feats, (train_c["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator("sigmoid", train_probs[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return train, val, 1 - cal(train_probs[:, 0]), 1 - cal(val_probs[:, 0])


def fit_s2_coral(df_all, feat_cols, s2_feats, use_coral=False, ttime_anchor=None):
    df_norm = apply_subject_z(df_all, feat_cols)
    if ttime_anchor is not None:
        # Anchor-based recentering: subtract per-subject mean of top-12 predicted NoPain
        df_norm = df_norm.copy()
        for subj, anchor_ids in ttime_anchor.items():
            mask = df_norm["subject"] == subj
            sub = df_norm[mask]
            anchor_mask = sub["segment_id"].isin(anchor_ids)
            if anchor_mask.sum() == 0:
                continue
            mu = sub.loc[anchor_mask, s2_feats].mean()
            df_norm.loc[mask, s2_feats] = (df_norm.loc[mask, s2_feats] - mu).astype(np.float32)
    train_full = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val_full = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    spec = ModelSpec("logreg", {"C": 1.0})
    pain_train = train_full[train_full["class"].isin(ARM_HAND)].reset_index(drop=True)
    train_probs = np.zeros((len(train_full), 2), dtype=np.float32)
    for subj in sorted(train_full["subject"].unique()):
        pain_tr = pain_train[pain_train["subject"] != subj].reset_index(drop=True)
        te = train_full[train_full["subject"] == subj].reset_index(drop=True)
        if len(pain_tr) == 0:
            continue
        if use_coral:
            Xs = pain_tr[s2_feats].to_numpy(dtype=np.float32)
            Xt = te[s2_feats].to_numpy(dtype=np.float32)
            Xs_a, _ = coral_transform(Xs, Xt)
            pain_tr = pain_tr.copy()
            pain_tr[s2_feats] = Xs_a.astype(np.float32)
        p = fit_binary_proba(pain_tr, te, s2_feats, armhand_binary(pain_tr["class"]), "robust", spec)
        train_probs[train_full["subject"] == subj] = p
    if use_coral:
        Xs = pain_train[s2_feats].to_numpy(dtype=np.float32)
        Xt = val_full[s2_feats].to_numpy(dtype=np.float32)
        Xs_a, _ = coral_transform(Xs, Xt)
        pt_c = pain_train.copy()
        pt_c[s2_feats] = Xs_a.astype(np.float32)
    else:
        pt_c = pain_train
    val_probs = fit_binary_proba(pt_c, val_full, s2_feats, armhand_binary(pt_c["class"]), "robust", spec)
    pain_mask = (train_full["class"].isin(ARM_HAND)).to_numpy()
    y_arm = (train_full.loc[pain_mask, "class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator("isotonic", train_probs[pain_mask, 0], y_arm)
    return train_full, val_full, cal(train_probs[:, 0]).astype(np.float32), cal(val_probs[:, 0]).astype(np.float32)


def predicted_nopain_anchors(df_sub, pain_prob):
    """Pick 12 lowest-pain-prob segments per subject = predicted NoPain."""
    anchors = {}
    nopain_score = 1 - pain_prob
    for subj in df_sub["subject"].unique():
        mask = (df_sub["subject"] == subj).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-nopain_score[mask])]
        top12 = df_sub.iloc[order[:12]]["segment_id"].tolist()
        anchors[int(subj)] = top12
    return anchors


def decode(df, pain_p, arm_p, w=(1, 1, 1)):
    s1 = np.column_stack([1 - pain_p, pain_p]).astype(np.float32)
    s2 = np.column_stack([arm_p, 1 - arm_p]).astype(np.float32)
    y_pred = np.zeros(len(df), dtype=int)
    for subj in sorted(df["subject"].unique()):
        mask = (df["subject"] == subj).to_numpy()
        y_pred[mask] = decode_joint_weighted(s1[mask], s2[mask], *w)
    return y_pred


def align(src_df, vals, ref_df):
    mp = dict(zip(src_df["segment_id"], vals))
    return np.array([mp[sid] for sid in ref_df["segment_id"]], dtype=np.float32)


def stats_per_subj(df, y_true, y_pred):
    arr = []
    for subj in sorted(df["subject"].unique()):
        mask = (df["subject"] == subj).to_numpy()
        arr.append(metrics_multiclass(y_true[mask], y_pred[mask])["macro_f1"])
    arr = np.array(arr)
    return {"mean": arr.mean(), "std": arr.std(ddof=1), "min": arr.min(), "max": arr.max(), "n_below_05": int((arr < 0.5).sum())}


def main() -> None:
    out_dir = Path("results/final/adaptation")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    resp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "resp"])
    s1_feats = bvp + eda
    s2_feats = resp

    variants = [
        ("baseline", False, False),
        ("coral_s1", True, False),
        ("coral_s2", False, True),
        ("coral_both", True, True),
    ]
    results = []
    for name, c1, c2 in variants:
        print(f"\n>>> {name} (coral_s1={c1} coral_s2={c2})")
        train_s1, val_s1, train_pain, val_pain = fit_s1_coral(df_all, feat_cols, s1_feats, use_coral=c1)
        train_s2, val_s2, train_arm, val_arm = fit_s2_coral(df_all, feat_cols, s2_feats, use_coral=c2)
        v_arm = align(val_s2, val_arm, val_s1)
        t_arm = align(train_s2, train_arm, train_s1)
        y_true_v = class_codes_3(val_s1["class"])
        y_true_t = class_codes_3(train_s1["class"])
        y_pred_v = decode(val_s1, val_pain, v_arm)
        y_pred_t = decode(train_s1, train_pain, t_arm)
        loso = stats_per_subj(train_s1, y_true_t, y_pred_t)
        val_ = stats_per_subj(val_s1, y_true_v, y_pred_v)
        print(f"  LOSO mean={loso['mean']:.4f} std={loso['std']:.4f} n<0.5={loso['n_below_05']}  |  VAL mean={val_['mean']:.4f} std={val_['std']:.4f}")
        results.append({"variant": name, **{f"loso_{k}": v for k, v in loso.items()}, **{f"val_{k}": v for k, v in val_.items()}})

    # Test-time anchor using baseline's stage1 predictions
    print(f"\n>>> ttime_anchor (use baseline s1 preds for anchor recentering)")
    train_s1, val_s1, train_pain, val_pain = fit_s1_coral(df_all, feat_cols, s1_feats, use_coral=False)
    t_anchors = predicted_nopain_anchors(train_s1, train_pain)
    v_anchors = predicted_nopain_anchors(val_s1, val_pain)
    all_anchors = {**t_anchors, **v_anchors}
    train_s2, val_s2, train_arm, val_arm = fit_s2_coral(df_all, feat_cols, s2_feats, use_coral=False, ttime_anchor=all_anchors)
    v_arm = align(val_s2, val_arm, val_s1)
    t_arm = align(train_s2, train_arm, train_s1)
    y_true_v = class_codes_3(val_s1["class"])
    y_true_t = class_codes_3(train_s1["class"])
    y_pred_v = decode(val_s1, val_pain, v_arm)
    y_pred_t = decode(train_s1, train_pain, t_arm)
    loso = stats_per_subj(train_s1, y_true_t, y_pred_t)
    val_ = stats_per_subj(val_s1, y_true_v, y_pred_v)
    print(f"  LOSO mean={loso['mean']:.4f} std={loso['std']:.4f} n<0.5={loso['n_below_05']}  |  VAL mean={val_['mean']:.4f} std={val_['std']:.4f}")
    results.append({"variant": "ttime_anchor", **{f"loso_{k}": v for k, v in loso.items()}, **{f"val_{k}": v for k, v in val_.items()}})

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== RANKING (by loso_mean) ===")
    print(df_res.sort_values("loso_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
