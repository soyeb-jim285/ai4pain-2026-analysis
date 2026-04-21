"""Ensemble CNN stage1 with classical XGB stage1 + suite31 stage2 logreg."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, apply_subject_z, armhand_binary,
    channel_of, class_codes_3, decode_joint_weighted, fit_binary_calibrator,
    fit_binary_proba, load_clean_features, metrics_multiclass, plot_confusion,
    unique_by_canonical,
)

CNN_DIR = Path("results/final/stage1_cnn_kaggle")
OUT = Path("results/final/ensemble")
OUT.mkdir(parents=True, exist_ok=True)


def fit_s1_xgb(df_all, feat_cols):
    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    s1_feats = bvp + eda
    dn = apply_subject_robust(df_all, feat_cols)
    train = dn[dn["split"] == "train"].reset_index(drop=True)
    val = dn[dn["split"] == "validation"].reset_index(drop=True)
    spec = ModelSpec("xgb", {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    tp = np.zeros((len(train), 2), dtype=np.float32)
    for s in sorted(train["subject"].unique()):
        tr = train[train["subject"] != s].reset_index(drop=True)
        te = train[train["subject"] == s].reset_index(drop=True)
        tp[train["subject"] == s] = fit_binary_proba(tr, te, s1_feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    vp = fit_binary_proba(train, val, s1_feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator("sigmoid", tp[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return train, val, 1 - cal(tp[:, 0]), 1 - cal(vp[:, 0])


def fit_s2_logreg(df_all, feat_cols):
    resp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "resp"])
    dn = apply_subject_z(df_all, feat_cols)
    tf = dn[dn["split"] == "train"].reset_index(drop=True)
    vf = dn[dn["split"] == "validation"].reset_index(drop=True)
    spec = ModelSpec("logreg", {"C": 1.0})
    pt = tf[tf["class"].isin(ARM_HAND)].reset_index(drop=True)
    tp = np.zeros((len(tf), 2), dtype=np.float32)
    for s in sorted(tf["subject"].unique()):
        pr = pt[pt["subject"] != s].reset_index(drop=True)
        te = tf[tf["subject"] == s].reset_index(drop=True)
        if len(pr) == 0:
            continue
        tp[tf["subject"] == s] = fit_binary_proba(pr, te, resp, armhand_binary(pr["class"]), "robust", spec)
    vp = fit_binary_proba(pt, vf, resp, armhand_binary(pt["class"]), "robust", spec)
    pm = (tf["class"].isin(ARM_HAND)).to_numpy()
    y = (tf.loc[pm, "class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator("isotonic", tp[pm, 0], y)
    return tf, vf, cal(tp[:, 0]).astype(np.float32), cal(vp[:, 0]).astype(np.float32)


def decode(df, pp, ap, w=(1, 1, 1)):
    s1 = np.column_stack([1 - pp, pp]).astype(np.float32)
    s2 = np.column_stack([ap, 1 - ap]).astype(np.float32)
    y = np.zeros(len(df), dtype=int)
    for s in sorted(df["subject"].unique()):
        m = (df["subject"] == s).to_numpy()
        y[m] = decode_joint_weighted(s1[m], s2[m], *w)
    return y


def align(src, vals, ref):
    mp = dict(zip(src["segment_id"], vals))
    return np.array([mp[sid] for sid in ref["segment_id"]], dtype=np.float32)


def stats_ps(df, yt, yp):
    arr = []
    for s in sorted(df["subject"].unique()):
        m = (df["subject"] == s).to_numpy()
        arr.append(metrics_multiclass(yt[m], yp[m])["macro_f1"])
    arr = np.array(arr)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
            "min": float(arr.min()), "max": float(arr.max()),
            "n_below_05": int((arr < 0.5).sum())}


def main():
    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")

    # Stage 1 XGB
    tr_s1, va_s1, xgb_pain_t, xgb_pain_v = fit_s1_xgb(df_all, feat_cols)

    # Stage 1 CNN probs (already computed on Kaggle)
    cnn_loso = pd.read_parquet(CNN_DIR / "loso_pain_probs.parquet")
    cnn_val = pd.read_parquet(CNN_DIR / "val_pain_probs.parquet")
    cnn_pain_t = align(cnn_loso, cnn_loso["pain_prob_cnn"].to_numpy(), tr_s1)
    cnn_pain_v = align(cnn_val, cnn_val["pain_prob_cnn"].to_numpy(), va_s1)

    # Stage 2 classical logreg
    tr_s2, va_s2, arm_t, arm_v = fit_s2_logreg(df_all, feat_cols)
    arm_t_al = align(tr_s2, arm_t, tr_s1)
    arm_v_al = align(va_s2, arm_v, va_s1)

    ytt = class_codes_3(tr_s1["class"])
    ytv = class_codes_3(va_s1["class"])

    results = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # alpha*CNN + (1-alpha)*XGB
        pt = alpha * cnn_pain_t + (1 - alpha) * xgb_pain_t
        pv = alpha * cnn_pain_v + (1 - alpha) * xgb_pain_v
        ypt = decode(tr_s1, pt, arm_t_al)
        ypv = decode(va_s1, pv, arm_v_al)
        loso = stats_ps(tr_s1, ytt, ypt)
        val_ = stats_ps(va_s1, ytv, ypv)
        results.append({"alpha_cnn": alpha, **{f"loso_{k}": v for k, v in loso.items()}, **{f"val_{k}": v for k, v in val_.items()}})
        print(f"alpha={alpha:.2f}  LOSO mean={loso['mean']:.4f} std={loso['std']:.4f}  |  VAL mean={val_['mean']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "ensemble_sweep.csv", index=False)
    print("\n=== BY VAL ===")
    print(df.sort_values("val_mean", ascending=False).to_string(index=False))
    print("\n=== BY LOSO STD ===")
    print(df.sort_values("loso_std").to_string(index=False))


if __name__ == "__main__":
    main()
