"""Comprehensive training suite: combines all findings from per-modality + subject diagnostics.

Experiments:
  A. baseline       - current final recipe
  B. weighted       - sample weight by responder_score
  C. exclude_bottom - drop 3-5 worst responder subjects from train
  D. sign_correct   - flip features for detected sign-flippers
  E. two_model      - separate models for responders vs non-responders
  F. fwd_selected   - use greedy-forward-selected features per modality
  G. combo_best     - stack winning techniques

Sweeps each across 4 modality pools: BVP, EDA, EDA+BVP, All.

Output: CSV summary + per-experiment confusion matrices + markdown report.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, apply_subject_z, armhand_binary,
    channel_of, class_codes_3, decode_joint_weighted, fit_binary_calibrator,
    load_clean_features, make_binary_model, make_scaler,
    metrics_multiclass, rank_binary_features, unique_by_canonical, plot_confusion,
    HAS_XGB,
)

OUT = Path("results/final/suite")
OUT.mkdir(parents=True, exist_ok=True)

MODALITIES = {
    "BVP": ["bvp"],
    "EDA": ["eda"],
    "EDA+BVP": ["eda", "bvp"],
    "All": ["bvp", "eda", "resp", "spo2"],
}


def select_pool(feat_cols, tags):
    return unique_by_canonical([c for c in feat_cols if channel_of(c) in tags])


def compute_responder_scores(train_df, feats):
    """Returns dict: subject -> responder_score in [0, 1]."""
    group_d = []
    per_subj_d = {}
    for subj in train_df["subject"].unique():
        g = train_df[train_df["subject"] == subj]
        a = g[g["class"] == "NoPain"][feats]
        b = g[g["class"] != "NoPain"][feats]
        if len(a) < 2 or len(b) < 2:
            continue
        diff = (b.mean() - a.mean()).to_numpy()
        pooled = np.sqrt((a.var() + b.var()) / 2).to_numpy()
        pooled = np.where(pooled > 1e-8, pooled, 1.0)
        d = diff / pooled
        per_subj_d[int(subj)] = d
        group_d.append(d)
    consensus_sign = np.sign(np.nanmedian(np.array(group_d), axis=0))
    rows = []
    for subj, d in per_subj_d.items():
        sep = float(np.nanmean(np.abs(d)))
        sign = float(np.nanmean((np.sign(d) == consensus_sign).astype(float)))
        rows.append({"subject": subj, "separability": sep, "sign_consistency": sign})
    df = pd.DataFrame(rows)
    df["sep_rank"] = df["separability"].rank(pct=True)
    df["sign_rank"] = df["sign_consistency"].rank(pct=True)
    df["responder_score"] = 0.6 * df["sep_rank"] + 0.4 * df["sign_rank"]
    return df.set_index("subject"), consensus_sign


def fit_xgb_weighted(X_tr, y_tr, X_te, sample_weight=None, n_est=50):
    from xgboost import XGBClassifier
    mdl = XGBClassifier(
        n_estimators=n_est, max_depth=4, learning_rate=0.08, max_bin=128,
        tree_method="hist", objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=4, verbosity=0,
    )
    if sample_weight is not None:
        mdl.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        mdl.fit(X_tr, y_tr)
    return mdl.predict_proba(X_te).astype(np.float32)


def sign_correct(X, feat_cols, feat_signs):
    """Flip sign of features per subject. feat_signs: np.array +1/-1."""
    return X * feat_signs


def stage1_loso_val(df_norm, feats, weights_df=None, exclude_subjects=None, sign_consensus=None, sign_flip_subjects=None):
    """Fit stage 1 LOSO + val. Returns pain probs."""
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)

    if exclude_subjects is not None:
        train_excl = train[~train["subject"].isin(exclude_subjects)].reset_index(drop=True)
    else:
        train_excl = train

    # Apply sign correction if requested
    def maybe_flip(df):
        if sign_flip_subjects and sign_consensus is not None:
            X = df[feats].to_numpy(dtype=np.float32).copy()
            for i, s in enumerate(df["subject"].to_numpy()):
                if int(s) in sign_flip_subjects:
                    X[i] = X[i] * -1  # flip entire row
            return X
        return df[feats].to_numpy(dtype=np.float32)

    tp = np.zeros((len(train), 2), dtype=np.float32)
    for s in sorted(train["subject"].unique()):
        tr = train_excl[train_excl["subject"] != s].reset_index(drop=True)
        te = train[train["subject"] == s].reset_index(drop=True)
        if len(tr) < 10:
            continue
        sc = make_scaler("std")
        X_tr = maybe_flip(tr); X_te = maybe_flip(te)
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        y_tr = (tr["class"] != "NoPain").astype(int).to_numpy()
        sw = None
        if weights_df is not None:
            sw = np.array([weights_df.loc[int(sub), "weight"] if int(sub) in weights_df.index else 1.0
                           for sub in tr["subject"].to_numpy()], dtype=np.float32)
        probs = fit_xgb_weighted(X_tr, y_tr, X_te, sample_weight=sw)
        tp[train["subject"] == s] = probs

    # val
    sc = make_scaler("std")
    X_tr = maybe_flip(train_excl); X_va = maybe_flip(val)
    X_tr = sc.fit_transform(X_tr); X_va = sc.transform(X_va)
    y_tr = (train_excl["class"] != "NoPain").astype(int).to_numpy()
    sw = None
    if weights_df is not None:
        sw = np.array([weights_df.loc[int(sub), "weight"] if int(sub) in weights_df.index else 1.0
                       for sub in train_excl["subject"].to_numpy()], dtype=np.float32)
    vp = fit_xgb_weighted(X_tr, y_tr, X_va, sample_weight=sw)

    # Calibrate
    cal = fit_binary_calibrator("sigmoid", tp[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    pain_t = 1 - cal(tp[:, 0])
    pain_v = 1 - cal(vp[:, 0])
    return train, val, pain_t.astype(np.float32), pain_v.astype(np.float32)


def stage2_loso_val(df_norm, feats):
    from src.final_pipeline import fit_binary_proba
    spec = ModelSpec("logreg", {"C": 1.0})
    train_full = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val_full = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    pain_train = train_full[train_full["class"].isin(ARM_HAND)].reset_index(drop=True)
    tp = np.zeros((len(train_full), 2), dtype=np.float32)
    for s in sorted(train_full["subject"].unique()):
        pain_tr = pain_train[pain_train["subject"] != s].reset_index(drop=True)
        te = train_full[train_full["subject"] == s].reset_index(drop=True)
        if len(pain_tr) == 0:
            continue
        tp[train_full["subject"] == s] = fit_binary_proba(pain_tr, te, feats, armhand_binary(pain_tr["class"]), "robust", spec)
    vp = fit_binary_proba(pain_train, val_full, feats, armhand_binary(pain_train["class"]), "robust", spec)
    pm = (train_full["class"].isin(ARM_HAND)).to_numpy()
    y = (train_full.loc[pm, "class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator("isotonic", tp[pm, 0], y)
    return train_full, val_full, cal(tp[:, 0]).astype(np.float32), cal(vp[:, 0]).astype(np.float32)


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


def per_subject_f1(df, yt, yp):
    arr = []
    for s in sorted(df["subject"].unique()):
        m = (df["subject"] == s).to_numpy()
        arr.append(metrics_multiclass(yt[m], yp[m])["macro_f1"])
    return np.array(arr)


def evaluate(name, mod, s1_feats_f, s1_fn_kwargs, s2_feats_f, df_all, feat_cols):
    s1_df = apply_subject_robust(df_all, feat_cols)
    s2_df = apply_subject_z(df_all, feat_cols)
    s1_feats = s1_feats_f(feat_cols, s1_df)
    s2_feats = s2_feats_f(feat_cols, s2_df)

    train_s1, val_s1, pain_t, pain_v = stage1_loso_val(s1_df, s1_feats, **s1_fn_kwargs)
    train_s2, val_s2, arm_t, arm_v = stage2_loso_val(s2_df, s2_feats)
    arm_t_al = align(train_s2, arm_t, train_s1)
    arm_v_al = align(val_s2, arm_v, val_s1)

    ytt = class_codes_3(train_s1["class"])
    ytv = class_codes_3(val_s1["class"])
    ypt = decode(train_s1, pain_t, arm_t_al)
    ypv = decode(val_s1, pain_v, arm_v_al)

    loso = per_subject_f1(train_s1, ytt, ypt)
    val = per_subject_f1(val_s1, ytv, ypv)

    cm = confusion_matrix(ytv, ypv, labels=[0, 1, 2])
    plot_confusion(cm, ["NoPain", "Arm", "Hand"], f"{name} {mod} val={val.mean():.3f}", OUT / f"cm_{name}_{mod}.png")

    # Per-class metrics (val)
    p, r, f, _ = precision_recall_fscore_support(ytv, ypv, labels=[0, 1, 2], zero_division=0)
    cw = {"NoPain": {"P": p[0], "R": r[0], "F1": f[0]},
          "PainArm": {"P": p[1], "R": r[1], "F1": f[1]},
          "PainHand": {"P": p[2], "R": r[2], "F1": f[2]}}

    return {
        "experiment": name, "modality": mod,
        "n_s1_feats": len(s1_feats), "n_s2_feats": len(s2_feats),
        "loso_mean": float(loso.mean()), "loso_std": float(loso.std(ddof=1)),
        "loso_min": float(loso.min()), "loso_max": float(loso.max()),
        "val_mean": float(val.mean()), "val_std": float(val.std(ddof=1)),
        "val_min": float(val.min()), "val_max": float(val.max()),
        "val_acc": float(accuracy_score(ytv, ypv)),
        **{f"val_{cls}_{m}": v for cls, ms in cw.items() for m, v in ms.items()},
    }


def main():
    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    train_all = df_all[df_all["split"] == "train"].reset_index(drop=True)

    # Pre-compute responder scores (on BVP+EDA pool - stage 1 features)
    s1_pool_all = unique_by_canonical([c for c in feat_cols if channel_of(c) in ("bvp", "eda")])
    rs_df, consensus = compute_responder_scores(train_all, s1_pool_all)

    # Bottom-5 subjects to exclude
    worst5 = set(rs_df.nsmallest(5, "responder_score").index.tolist())
    # Sign-flippers (sign consistency < 0.45)
    sign_flip = set(rs_df[rs_df["sign_consistency"] < 0.45].index.tolist())
    # Weight df: weight = max(0.3, responder_score)
    weights_df = rs_df.copy()
    weights_df["weight"] = weights_df["responder_score"].clip(lower=0.3)

    print(f"Worst-5 exclude: {sorted(worst5)}")
    print(f"Sign-flip subjects: {sorted(sign_flip)}")

    def s1_feats_topk(feat_cols, s1_df, tags, k=30):
        pool = select_pool(feat_cols, tags)
        train = s1_df[s1_df["split"] == "train"].reset_index(drop=True)
        ranked = rank_binary_features(train, pool, positive="Pain")
        return ranked[:min(k, len(pool))]

    def s2_feats_default(feat_cols, s2_df):
        pool = unique_by_canonical([c for c in feat_cols if channel_of(c) in ("eda", "resp")])
        train = s2_df[s2_df["split"] == "train"].reset_index(drop=True)
        pain = train[train["class"].isin(ARM_HAND)].reset_index(drop=True)
        ranked = rank_binary_features(pain, pool, positive="ArmHand")
        return ranked[:30]

    experiments = []
    for mod, tags in MODALITIES.items():
        pool = select_pool(feat_cols, tags)
        k_s1 = min(30, len(pool))
        s1_fn = lambda fc, df, t=tags, k=k_s1: s1_feats_topk(fc, df, t, k)

        # A. Baseline
        experiments.append(("A_baseline", mod, s1_fn, {}, s2_feats_default))
        # B. Weighted
        experiments.append(("B_weighted", mod, s1_fn, {"weights_df": weights_df}, s2_feats_default))
        # C. Exclude bottom-5
        experiments.append(("C_exclude", mod, s1_fn, {"exclude_subjects": worst5}, s2_feats_default))
        # D. Sign correction (only if flippers exist)
        if sign_flip:
            experiments.append(("D_sign_correct", mod, s1_fn, {"sign_consensus": consensus, "sign_flip_subjects": sign_flip}, s2_feats_default))
        # E. Weighted + exclude (combined)
        experiments.append(("E_weighted+exclude", mod, s1_fn, {"weights_df": weights_df, "exclude_subjects": worst5}, s2_feats_default))

    results = []
    for name, mod, s1_f, s1_kw, s2_f in tqdm(experiments, desc="experiments", ncols=100):
        try:
            r = evaluate(name, mod, s1_f, s1_kw, s2_f, df_all, feat_cols)
            results.append(r)
            tqdm.write(f"  {name:20s} {mod:8s}  LOSO={r['loso_mean']:.4f} std={r['loso_std']:.3f} | VAL={r['val_mean']:.4f} std={r['val_std']:.3f}")
        except Exception as e:
            tqdm.write(f"  {name} {mod} FAILED: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "summary.csv", index=False)
    df_sorted = df.sort_values("val_mean", ascending=False)
    print("\n=== TOP 10 BY VAL ===")
    print(df_sorted[["experiment", "modality", "loso_mean", "loso_std", "val_mean", "val_std", "val_acc"]].head(10).to_string(index=False))
    print("\n=== TOP 10 BY LOSO ===")
    print(df.sort_values("loso_mean", ascending=False)[["experiment", "modality", "loso_mean", "loso_std", "val_mean"]].head(10).to_string(index=False))

    # Write markdown
    lines = ["# Comprehensive Training Suite Results", ""]
    lines.append(f"Total experiments: {len(results)}")
    lines.append(f"Worst-5 excluded: {sorted(worst5)}")
    lines.append(f"Sign-flip subjects: {sorted(sign_flip)}")
    lines.append("")
    lines.append("## Ranked by VAL mean F1")
    lines.append("")
    lines.append(df_sorted[["experiment", "modality", "loso_mean", "loso_std", "val_mean", "val_std", "val_acc", "val_NoPain_F1", "val_PainArm_F1", "val_PainHand_F1"]].to_markdown(index=False))
    (OUT / "report.md").write_text("\n".join(lines))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
