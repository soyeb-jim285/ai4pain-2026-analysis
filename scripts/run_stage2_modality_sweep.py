"""Stage 2 modality sweep: find best feature set for arm vs hand."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, apply_subject_z, armhand_binary,
    channel_of, class_codes_3, decode_joint_weighted, fit_binary_calibrator,
    fit_binary_proba, load_clean_features, metrics_multiclass, unique_by_canonical,
)


def rank_features(train, feats, pos="ArmHand"):
    vp, vn = np.zeros(len(feats)), np.zeros(len(feats))
    mags = []
    for subj in train["subject"].unique():
        g = train[train["subject"] == subj]
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


def fit_s1(df_all, feat_cols):
    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    feats = bvp + eda
    dn = apply_subject_robust(df_all, feat_cols)
    train = dn[dn["split"] == "train"].reset_index(drop=True)
    val = dn[dn["split"] == "validation"].reset_index(drop=True)
    spec = ModelSpec("xgb", {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    tp = np.zeros((len(train), 2), dtype=np.float32)
    for s in sorted(train["subject"].unique()):
        tr = train[train["subject"] != s].reset_index(drop=True)
        te = train[train["subject"] == s].reset_index(drop=True)
        tp[train["subject"] == s] = fit_binary_proba(tr, te, feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    vp = fit_binary_proba(train, val, feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator("sigmoid", tp[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return train, val, 1 - cal(tp[:, 0]), 1 - cal(vp[:, 0])


def fit_s2(df_all, feat_cols, s2_feats, model_name="logreg"):
    dn = apply_subject_z(df_all, feat_cols)
    tf = dn[dn["split"] == "train"].reset_index(drop=True)
    vf = dn[dn["split"] == "validation"].reset_index(drop=True)
    spec = ModelSpec(model_name, {"C": 1.0})
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
    return tf, vf, cal(tp[:, 0]).astype(np.float32), cal(vp[:, 0]).astype(np.float32), vp


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
    out = Path("results/final/stage2_modality_sweep")
    out.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    train_all = df_all[df_all["split"] == "train"].reset_index(drop=True)
    pain_train = train_all[train_all["class"].isin(ARM_HAND)].reset_index(drop=True)

    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    resp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "resp"])
    spo2 = unique_by_canonical([c for c in feat_cols if channel_of(c) == "spo2"])
    print(f"BVP: {len(bvp)}  EDA: {len(eda)}  RESP: {len(resp)}  SpO2: {len(spo2)}")

    # Rank each modality for arm-vs-hand
    rank_all = rank_features(pain_train, bvp + eda + resp + spo2, "ArmHand")
    rank_resp = rank_features(pain_train, resp)
    rank_bvp = rank_features(pain_train, bvp)
    rank_eda = rank_features(pain_train, eda)
    rank_spo2 = rank_features(pain_train, spo2)
    rank_bes = rank_features(pain_train, bvp + eda + spo2)

    # Fit stage 1 once (same across all stage2 variants)
    train_s1, val_s1, pain_t, pain_v = fit_s1(df_all, feat_cols)
    ytt = class_codes_3(train_s1["class"])
    ytv = class_codes_3(val_s1["class"])

    variants = [
        ("resp_all", resp),
        ("resp_top20", rank_resp.head(20).index.tolist()),
        ("bvp_only_top20", rank_bvp.head(20).index.tolist()),
        ("eda_only_top20", rank_eda.head(20).index.tolist()),
        ("spo2_only_top20", rank_spo2.head(20).index.tolist()),
        ("bvp_eda_top30", rank_features(pain_train, bvp + eda).head(30).index.tolist()),
        ("bvp_resp_top30", rank_features(pain_train, bvp + resp).head(30).index.tolist()),
        ("eda_resp_top30", rank_features(pain_train, eda + resp).head(30).index.tolist()),
        ("all_top20", rank_all.head(20).index.tolist()),
        ("all_top40", rank_all.head(40).index.tolist()),
        ("all_top80", rank_all.head(80).index.tolist()),
        ("bes_top30", rank_bes.head(30).index.tolist()),
    ]

    results = []
    for name, s2_feats in tqdm(variants, desc="sweep"):
        train_s2, val_s2, arm_t, arm_v, val_raw = fit_s2(df_all, feat_cols, s2_feats, "logreg")
        t_al = align(train_s2, arm_t, train_s1)
        v_al = align(val_s2, arm_v, val_s1)

        # Stage 2 AUC (arm vs hand on pain-only val rows)
        pain_mask_va = val_s2["class"].isin(ARM_HAND).to_numpy()
        y_arm_va = (val_s2.loc[pain_mask_va, "class"] == "PainArm").astype(int).to_numpy()
        arm_prob_pain = arm_v[pain_mask_va]
        s2_auc = roc_auc_score(y_arm_va, arm_prob_pain)

        ypt = decode(train_s1, pain_t, t_al)
        ypv = decode(val_s1, pain_v, v_al)
        loso = stats_ps(train_s1, ytt, ypt)
        val_ = stats_ps(val_s1, ytv, ypv)
        results.append({
            "variant": name, "n_feats": len(s2_feats), "s2_auc": s2_auc,
            "loso_mean": loso["mean"], "loso_std": loso["std"],
            "val_mean": val_["mean"], "val_std": val_["std"],
        })
        print(f"  {name:25s} n={len(s2_feats):3d}  s2_AUC={s2_auc:.3f}  LOSO={loso['mean']:.4f}  VAL={val_['mean']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(out / "summary.csv", index=False)
    print("\n=== BY S2 AUC ===")
    print(df.sort_values("s2_auc", ascending=False).to_string(index=False))
    print("\n=== BY VAL F1 ===")
    print(df.sort_values("val_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
