"""Preprocessing variants to reduce LOSO variance."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND, ModelSpec, armhand_binary, class_codes_3,
    decode_joint_weighted, fit_binary_calibrator, fit_binary_proba,
    load_clean_features, metrics_multiclass,
    stage1_feature_sets, stage2_feature_sets,
)


def apply_rank_per_subject(df, feat_cols):
    """Replace features with percentile rank within each subject."""
    out = df.copy()
    for subj in out["subject"].unique():
        mask = out["subject"] == subj
        grp = out.loc[mask, feat_cols]
        ranked = grp.rank(method="average", pct=True)
        out.loc[mask, feat_cols] = ranked.astype(np.float32)
    return out


def apply_trimmed_z(df, feat_cols, trim=0.1):
    """Subject z-score using trimmed mean/std (robust to outliers)."""
    out = df.copy()
    for subj in out["subject"].unique():
        mask = out["subject"] == subj
        X = out.loc[mask, feat_cols].to_numpy(dtype=np.float32)
        lo = np.quantile(X, trim, axis=0)
        hi = np.quantile(X, 1 - trim, axis=0)
        Xt = X.copy()
        for j in range(X.shape[1]):
            keep = (X[:, j] >= lo[j]) & (X[:, j] <= hi[j])
            if keep.sum() > 1:
                mu = X[keep, j].mean()
                sd = X[keep, j].std()
            else:
                mu, sd = X[:, j].mean(), X[:, j].std()
            sd = sd if sd > 1e-8 else 1.0
            Xt[:, j] = (X[:, j] - mu) / sd
        out.loc[mask, feat_cols] = Xt.astype(np.float32)
    return out


def apply_subject_clip_z(df, feat_cols, clip=3.0):
    """Subject-z then clip extreme values."""
    out = df.copy()
    means = out.groupby("subject")[feat_cols].transform("mean")
    stds = out.groupby("subject")[feat_cols].transform("std", ddof=0).where(lambda s: s > 0, 1.0)
    out[feat_cols] = ((out[feat_cols] - means) / stds).clip(-clip, clip).fillna(0).astype(np.float32)
    return out


def apply_quantile_map(df, feat_cols, n_q=10):
    """Per-subject quantile bucketing to n_q equal bins."""
    out = df.copy()
    for subj in out["subject"].unique():
        mask = out["subject"] == subj
        X = out.loc[mask, feat_cols]
        binned = X.apply(lambda s: pd.qcut(s, q=min(n_q, s.nunique()), labels=False, duplicates="drop"), axis=0)
        out.loc[mask, feat_cols] = binned.fillna(0).astype(np.float32).to_numpy()
    return out


def apply_subject_z(df, feat_cols):
    out = df.copy()
    means = out.groupby("subject")[feat_cols].transform("mean")
    stds = out.groupby("subject")[feat_cols].transform("std", ddof=0).where(lambda s: s > 0, 1.0)
    out[feat_cols] = ((out[feat_cols] - means) / stds).fillna(0).astype(np.float32)
    return out


def apply_subject_robust(df, feat_cols):
    out = df.copy()
    med = out.groupby("subject")[feat_cols].transform("median")
    q75 = out.groupby("subject")[feat_cols].transform(lambda s: s.quantile(0.75))
    q25 = out.groupby("subject")[feat_cols].transform(lambda s: s.quantile(0.25))
    iqr = (q75 - q25).where(lambda s: s > 0, 1.0)
    out[feat_cols] = ((out[feat_cols] - med) / iqr).fillna(0).astype(np.float32)
    return out


PREPROC = {
    "subject_z": apply_subject_z,
    "subject_robust": apply_subject_robust,
    "rank": apply_rank_per_subject,
    "trimmed_z": apply_trimmed_z,
    "clip_z": apply_subject_clip_z,
    "quantile": apply_quantile_map,
}


def fit_s1(df_all, feat_cols, preproc_fn):
    df_norm = preproc_fn(df_all, feat_cols)
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


def fit_s2(df_all, feat_cols, preproc_fn):
    df_norm = preproc_fn(df_all, feat_cols)
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


def per_subject_stats(df, y_true, y_pred):
    rows = []
    for subj in sorted(df["subject"].unique()):
        mask = (df["subject"] == subj).to_numpy()
        rows.append(metrics_multiclass(y_true[mask], y_pred[mask])["macro_f1"])
    arr = np.array(rows)
    return {"mean": arr.mean(), "std": arr.std(ddof=1), "min": arr.min(), "max": arr.max(), "n_below_05": int((arr < 0.5).sum())}


def main() -> None:
    out_dir = Path("results/final/preproc_loso")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")

    # Test: vary stage1 preproc, keep stage2 at subject_z baseline
    preproc_variants = ["subject_z", "subject_robust", "rank", "trimmed_z", "clip_z"]
    results = []
    for s1_pp in preproc_variants:
        print(f"\n>>> s1_preproc={s1_pp}")
        try:
            train_s1, val_s1, train_pain, val_pain = fit_s1(df_all, feat_cols, PREPROC[s1_pp])
            train_s2, val_s2, train_arm, val_arm = fit_s2(df_all, feat_cols, PREPROC["subject_z"])
            v_arm = align(val_s2, val_arm, val_s1)
            t_arm = align(train_s2, train_arm, train_s1)
            y_true_v = class_codes_3(val_s1["class"])
            y_true_t = class_codes_3(train_s1["class"])
            y_pred_v = decode(val_s1, val_pain, v_arm, 1, 1, 1)
            y_pred_t = decode(train_s1, train_pain, t_arm, 1, 1, 1)
            loso = per_subject_stats(train_s1, y_true_t, y_pred_t)
            val_ = per_subject_stats(val_s1, y_true_v, y_pred_v)
            print(f"  LOSO mean={loso['mean']:.4f} std={loso['std']:.4f} min={loso['min']:.3f} max={loso['max']:.3f} n<0.5={loso['n_below_05']}")
            print(f"  VAL  mean={val_['mean']:.4f} std={val_['std']:.4f} min={val_['min']:.3f} max={val_['max']:.3f} n<0.5={val_['n_below_05']}")
            results.append({
                "s1_preproc": s1_pp,
                "loso_mean": loso["mean"], "loso_std": loso["std"], "loso_min": loso["min"], "loso_max": loso["max"], "loso_n_below05": loso["n_below_05"],
                "val_mean": val_["mean"], "val_std": val_["std"], "val_min": val_["min"], "val_max": val_["max"], "val_n_below05": val_["n_below_05"],
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    df = pd.DataFrame(results).sort_values("loso_std")
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== RANKING (by loso_std, lowest first) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
