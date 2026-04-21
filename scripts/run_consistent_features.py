"""Feature selection: keep only features that consistently separate NoPain/Pain across subjects."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, apply_subject_z, armhand_binary,
    class_codes_3, decode_joint_weighted, fit_binary_calibrator, fit_binary_proba,
    load_clean_features, metrics_multiclass, stage1_feature_sets, stage2_feature_sets,
    channel_of, unique_by_canonical,
)


def rank_features_by_consistency(train, feat_cols, positive="Pain"):
    """Fraction of subjects where feature's NoPain mean < Pain mean (or consistent direction)."""
    votes_pos, votes_neg = np.zeros(len(feat_cols)), np.zeros(len(feat_cols))
    magnitudes = []
    for subj in train["subject"].unique():
        g = train[train["subject"] == subj]
        if positive == "Pain":
            a = g[g["class"] == "NoPain"][feat_cols]
            b = g[g["class"] != "NoPain"][feat_cols]
        else:
            a = g[g["class"] == "PainArm"][feat_cols]
            b = g[g["class"] == "PainHand"][feat_cols]
        if len(a) < 2 or len(b) < 2:
            continue
        diff = (b.mean() - a.mean()).to_numpy()
        pooled = np.sqrt((a.var() + b.var()) / 2).to_numpy()
        pooled = np.where(pooled > 1e-8, pooled, 1.0)
        d = diff / pooled
        votes_pos += (d > 0.1).astype(int)
        votes_neg += (d < -0.1).astype(int)
        magnitudes.append(np.abs(d))
    consistency = np.maximum(votes_pos, votes_neg) / (votes_pos + votes_neg + 1e-6)
    mag = np.mean(magnitudes, axis=0) if magnitudes else np.zeros(len(feat_cols))
    score = consistency * mag
    return pd.DataFrame({"feature": feat_cols, "consistency": consistency, "effect_size": mag, "score": score}).sort_values("score", ascending=False)


def fit_s1(df_norm, feats, ):
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
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


def fit_s2(df_norm, feats):
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
        p = fit_binary_proba(pain_tr, te, feats, armhand_binary(pain_tr["class"]), "robust", spec)
        train_probs[train_full["subject"] == subj] = p
    val_probs = fit_binary_proba(pain_train, val_full, feats, armhand_binary(pain_train["class"]), "robust", spec)
    pain_mask = (train_full["class"].isin(ARM_HAND)).to_numpy()
    y_arm = (train_full.loc[pain_mask, "class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator("isotonic", train_probs[pain_mask, 0], y_arm)
    return train_full, val_full, cal(train_probs[:, 0]).astype(np.float32), cal(val_probs[:, 0]).astype(np.float32)


def decode(df, pain_p, arm_p, w=(1, 1, 1)):
    s1 = np.column_stack([1 - pain_p, pain_p]).astype(np.float32)
    s2 = np.column_stack([arm_p, 1 - arm_p]).astype(np.float32)
    y_pred = np.zeros(len(df), dtype=int)
    for subj in sorted(df["subject"].unique()):
        mask = (df["subject"] == subj).to_numpy()
        y_pred[mask] = decode_joint_weighted(s1[mask], s2[mask], *w)
    return y_pred


def align(src_df, src_vals, ref_df):
    mp = dict(zip(src_df["segment_id"], src_vals))
    return np.array([mp[sid] for sid in ref_df["segment_id"]], dtype=np.float32)


def per_subject_stats(df, y_true, y_pred):
    arr = []
    for subj in sorted(df["subject"].unique()):
        mask = (df["subject"] == subj).to_numpy()
        arr.append(metrics_multiclass(y_true[mask], y_pred[mask])["macro_f1"])
    arr = np.array(arr)
    return {"mean": arr.mean(), "std": arr.std(ddof=1), "min": arr.min(), "max": arr.max(), "n_below_05": int((arr < 0.5).sum())}


def main() -> None:
    out_dir = Path("results/final/consistent_feats")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    train_all = df_all[df_all["split"] == "train"].reset_index(drop=True)

    # Rank stage1 features (pain separability)
    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    s1_pool = bvp + eda
    print(f"Stage1 pool: {len(s1_pool)} features")
    s1_rank = rank_features_by_consistency(train_all, s1_pool, "Pain")
    s1_rank.to_csv(out_dir / "s1_feature_ranking.csv", index=False)

    # Rank stage2 features (arm vs hand separability)
    resp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "resp"])
    train_pain = train_all[train_all["class"].isin(ARM_HAND)].reset_index(drop=True)
    s2_rank = rank_features_by_consistency(train_pain, resp, "ArmHand")
    s2_rank.to_csv(out_dir / "s2_feature_ranking.csv", index=False)

    # Test: top K features per stage
    s1_df_norm = apply_subject_robust(df_all, feat_cols)
    s2_df_norm = apply_subject_z(df_all, feat_cols)
    results = []
    for topk_s1 in [30, 50, 80, len(s1_pool)]:
        for topk_s2 in [20, 30, 44]:
            s1_feats = s1_rank["feature"].head(topk_s1).tolist()
            s2_feats = s2_rank["feature"].head(topk_s2).tolist()
            print(f"\n>>> s1_top={topk_s1} s2_top={topk_s2}")
            train_s1, val_s1, train_pain, val_pain = fit_s1(s1_df_norm, s1_feats)
            train_s2, val_s2, train_arm, val_arm = fit_s2(s2_df_norm, s2_feats)
            v_arm = align(val_s2, val_arm, val_s1)
            t_arm = align(train_s2, train_arm, train_s1)
            y_true_v = class_codes_3(val_s1["class"])
            y_true_t = class_codes_3(train_s1["class"])
            y_pred_v = decode(val_s1, val_pain, v_arm)
            y_pred_t = decode(train_s1, train_pain, t_arm)
            loso = per_subject_stats(train_s1, y_true_t, y_pred_t)
            val_ = per_subject_stats(val_s1, y_true_v, y_pred_v)
            print(f"  LOSO mean={loso['mean']:.4f} std={loso['std']:.4f} min={loso['min']:.3f} n<0.5={loso['n_below_05']}  |  VAL mean={val_['mean']:.4f} std={val_['std']:.4f}")
            results.append({
                "topk_s1": topk_s1, "topk_s2": topk_s2,
                "loso_mean": loso["mean"], "loso_std": loso["std"], "loso_min": loso["min"], "loso_n_below05": loso["n_below_05"],
                "val_mean": val_["mean"], "val_std": val_["std"], "val_min": val_["min"],
            })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== By LOSO mean desc ===")
    print(df.sort_values("loso_mean", ascending=False).to_string(index=False))
    print("\n=== By LOSO std asc ===")
    print(df.sort_values("loso_std").to_string(index=False))


if __name__ == "__main__":
    main()
