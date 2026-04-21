"""Final pipeline with overfit fix: XGB stage1 n_estimators=50 based on learning curve."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, apply_subject_z, armhand_binary,
    channel_of, class_codes_3, decode_joint_weighted, fit_binary_calibrator,
    fit_binary_proba, load_clean_features, metrics_multiclass,
    plot_confusion, plot_per_subject, unique_by_canonical,
)

RESAMPLE_PATHS = {
    "truncate1022": "results/tables/all_features_merged_1022.parquet",
    "linear1022": "results/tables/all_features_merged_linear1022.parquet",
    "poly1022": "results/tables/all_features_merged_poly1022.parquet",
}


def rank_features(train, feats, pos="Pain"):
    vp, vn = np.zeros(len(feats)), np.zeros(len(feats))
    mags = []
    for subj in train["subject"].unique():
        g = train[train["subject"] == subj]
        if pos == "Pain":
            a, b = g[g["class"] == "NoPain"][feats], g[g["class"] != "NoPain"][feats]
        else:
            a, b = g[g["class"] == "PainArm"][feats], g[g["class"] == "PainHand"][feats]
        if len(a) < 2 or len(b) < 2:
            continue
        diff = (b.mean() - a.mean()).to_numpy()
        pooled = np.sqrt((a.var() + b.var()) / 2).to_numpy()
        pooled = np.where(pooled > 1e-8, pooled, 1.0)
        d = diff / pooled
        vp += (d > 0.1).astype(int)
        vn += (d < -0.1).astype(int)
        mags.append(np.abs(d))
    cons = np.maximum(vp, vn) / (vp + vn + 1e-6)
    mag = np.mean(mags, axis=0) if mags else np.zeros(len(feats))
    return pd.Series(cons * mag, index=feats).sort_values(ascending=False)


def fit_s1(df_all, feat_cols, s1_feats, n_est=50):
    dn = apply_subject_robust(df_all, feat_cols)
    train = dn[dn["split"] == "train"].reset_index(drop=True)
    val = dn[dn["split"] == "validation"].reset_index(drop=True)
    spec = ModelSpec("xgb", {"n_estimators": n_est, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    tp = np.zeros((len(train), 2), dtype=np.float32)
    for s in sorted(train["subject"].unique()):
        tr = train[train["subject"] != s].reset_index(drop=True)
        te = train[train["subject"] == s].reset_index(drop=True)
        tp[train["subject"] == s] = fit_binary_proba(tr, te, s1_feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    vp = fit_binary_proba(train, val, s1_feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator("sigmoid", tp[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return train, val, 1 - cal(tp[:, 0]), 1 - cal(vp[:, 0])


def fit_s2(df_all, feat_cols, s2_feats):
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
        tp[tf["subject"] == s] = fit_binary_proba(pr, te, s2_feats, armhand_binary(pr["class"]), "robust", spec)
    vp = fit_binary_proba(pt, vf, s2_feats, armhand_binary(pt["class"]), "robust", spec)
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


def align(src, v, ref):
    mp = dict(zip(src["segment_id"], v))
    return np.array([mp[sid] for sid in ref["segment_id"]], dtype=np.float32)


def stats_ps(df, y_true, y_pred):
    arr = []
    for s in sorted(df["subject"].unique()):
        m = (df["subject"] == s).to_numpy()
        arr.append(metrics_multiclass(y_true[m], y_pred[m])["macro_f1"])
    arr = np.array(arr)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "min": float(arr.min()),
            "max": float(arr.max()), "n_below_05": int((arr < 0.5).sum())}, arr


def main():
    out = Path("results/final/final_tuned")
    out.mkdir(parents=True, exist_ok=True)

    df_ref, fc_ref = load_clean_features(RESAMPLE_PATHS["truncate1022"])
    tr_ref = df_ref[df_ref["split"] == "train"].reset_index(drop=True)
    bvp = unique_by_canonical([c for c in fc_ref if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in fc_ref if channel_of(c) == "eda"])
    resp = unique_by_canonical([c for c in fc_ref if channel_of(c) == "resp"])
    s1r = rank_features(tr_ref, bvp + eda, "Pain")
    s2r = rank_features(tr_ref[tr_ref["class"].isin(ARM_HAND)].reset_index(drop=True), resp, "ArmHand")

    variants = [
        ("n50_s1all_s2top20", 50, len(bvp + eda), 20),
        ("n50_s1top50_s2top20", 50, 50, 20),
        ("n100_s1all_s2top20", 100, len(bvp + eda), 20),
        ("n200_baseline", 200, len(bvp + eda), 20),  # for comparison
    ]

    results = []
    for vname, n_est, k1, k2 in variants:
        print(f"\n>>> {vname} (n_est={n_est}, s1_top={k1}, s2_top={k2})")
        s1f = s1r.head(k1).index.tolist()
        s2f = s2r.head(k2).index.tolist()
        # Multiview
        pt_l, at_l, pv_l, av_l = [], [], [], []
        rt, rv = None, None
        for view in ["truncate1022", "linear1022", "poly1022"]:
            df_all, fc = load_clean_features(RESAMPLE_PATHS[view])
            s1fv = [f for f in s1f if f in fc]
            s2fv = [f for f in s2f if f in fc]
            t1, v1, tp, vp = fit_s1(df_all, fc, s1fv, n_est=n_est)
            t2, v2, ta, va = fit_s2(df_all, fc, s2fv)
            if rt is None:
                rt, rv = t1, v1
                pt_l.append(tp); at_l.append(align(t2, ta, rt))
                pv_l.append(vp); av_l.append(align(v2, va, rv))
            else:
                pt_l.append(align(t1, tp, rt)); at_l.append(align(t2, ta, rt))
                pv_l.append(align(v1, vp, rv)); av_l.append(align(v2, va, rv))
        pt = np.mean(pt_l, axis=0); at = np.mean(at_l, axis=0)
        pv = np.mean(pv_l, axis=0); av = np.mean(av_l, axis=0)

        ytt = class_codes_3(rt["class"]); ytv = class_codes_3(rv["class"])
        ypt = decode(rt, pt, at); ypv = decode(rv, pv, av)
        loso, loso_arr = stats_ps(rt, ytt, ypt)
        val, val_arr = stats_ps(rv, ytv, ypv)
        print(f"  LOSO mean={loso['mean']:.4f} std={loso['std']:.4f} n<0.5={loso['n_below_05']}  |  VAL mean={val['mean']:.4f} std={val['std']:.4f}")
        results.append({
            "variant": vname, "n_estimators": n_est, "topk_s1": k1, "topk_s2": k2,
            **{f"loso_{k}": v for k, v in loso.items()},
            **{f"val_{k}": v for k, v in val.items()},
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(out / "summary.csv", index=False)
    print("\n=== RANKING (by loso_mean) ===")
    print(df_res.sort_values("loso_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
