"""Final combined pipeline - stacks all proven boosts.

Boosts:
1. Consistency-ranked feature selection (top K per stage)
2. Multiview ensemble (truncate + linear + poly)
3. Suite 31 stage2 recipe (logreg + isotonic)
4. Upgraded stage1 (bvp_eda_core + subject_robust + xgb + sigmoid)
5. Decoder weights (1,1,1) fixed + optional tune

Outputs per-subject LOSO/VAL stats + model save.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, apply_subject_z, armhand_binary,
    channel_of, class_codes_3, decode_joint_weighted, fit_binary_calibrator,
    fit_binary_proba, load_clean_features, make_binary_model, make_scaler,
    metrics_multiclass, plot_confusion, plot_per_subject, unique_by_canonical,
)

RESAMPLE_PATHS = {
    "truncate1022": "results/tables/all_features_merged_1022.parquet",
    "linear1022": "results/tables/all_features_merged_linear1022.parquet",
    "poly1022": "results/tables/all_features_merged_poly1022.parquet",
}


def rank_features(train, feat_cols, positive="Pain"):
    votes_pos, votes_neg = np.zeros(len(feat_cols)), np.zeros(len(feat_cols))
    mags = []
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
        mags.append(np.abs(d))
    consistency = np.maximum(votes_pos, votes_neg) / (votes_pos + votes_neg + 1e-6)
    mag = np.mean(mags, axis=0) if mags else np.zeros(len(feat_cols))
    score = consistency * mag
    return pd.Series(score, index=feat_cols).sort_values(ascending=False)


def fit_s1_one_view(df_all, feat_cols, s1_feats):
    df_norm = apply_subject_robust(df_all, feat_cols)
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    # Fixed n=50 matches EDA+BVP optimal; ES with tiny inner val hurts
    spec = ModelSpec("xgb", {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    for subj in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subj].reset_index(drop=True)
        te = train[train["subject"] == subj].reset_index(drop=True)
        p = fit_binary_proba(tr, te, s1_feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
        train_probs[train["subject"] == subj] = p
    val_probs = fit_binary_proba(train, val, s1_feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator("sigmoid", train_probs[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return train, val, 1 - cal(train_probs[:, 0]), 1 - cal(val_probs[:, 0]), spec, cal


def fit_s2_one_view(df_all, feat_cols, s2_feats):
    df_norm = apply_subject_z(df_all, feat_cols)
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
        p = fit_binary_proba(pain_tr, te, s2_feats, armhand_binary(pain_tr["class"]), "robust", spec)
        train_probs[train_full["subject"] == subj] = p
    val_probs = fit_binary_proba(pain_train, val_full, s2_feats, armhand_binary(pain_train["class"]), "robust", spec)
    pain_mask = (train_full["class"].isin(ARM_HAND)).to_numpy()
    y_arm = (train_full.loc[pain_mask, "class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator("isotonic", train_probs[pain_mask, 0], y_arm)
    return train_full, val_full, cal(train_probs[:, 0]).astype(np.float32), cal(val_probs[:, 0]).astype(np.float32), spec, cal


def decode_subjectwise(df, pain_p, arm_p, w=(1.0, 1.0, 1.0)):
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


def per_subject(df, y_true, y_pred):
    rows = []
    for subj in sorted(df["subject"].unique()):
        mask = (df["subject"] == subj).to_numpy()
        m = metrics_multiclass(y_true[mask], y_pred[mask])
        rows.append({"subject": int(subj), "f1": m["macro_f1"], "acc": m["accuracy"]})
    return pd.DataFrame(rows)


def stats(per_sub):
    arr = per_sub["f1"].to_numpy()
    return {
        "mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
        "min": float(arr.min()), "max": float(arr.max()),
        "n_below_05": int((arr < 0.5).sum()), "n_subjects": len(arr),
    }


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rank features once using truncate1022 (cleanest signal)
    print(">>> Ranking features for consistency...")
    df_ref, feat_cols_ref = load_clean_features(RESAMPLE_PATHS["truncate1022"])
    train_ref = df_ref[df_ref["split"] == "train"].reset_index(drop=True)
    bvp = unique_by_canonical([c for c in feat_cols_ref if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols_ref if channel_of(c) == "eda"])
    resp = unique_by_canonical([c for c in feat_cols_ref if channel_of(c) == "resp"])
    s1_rank = rank_features(train_ref, bvp + eda, "Pain")
    pain_ref = train_ref[train_ref["class"].isin(ARM_HAND)].reset_index(drop=True)
    if args.s2_modality == "eda_resp":
        s2_pool = eda + resp
    elif args.s2_modality == "bvp_resp":
        s2_pool = bvp + resp
    elif args.s2_modality == "all":
        s2_pool = bvp + eda + resp
    else:
        s2_pool = resp
    s2_rank = rank_features(pain_ref, s2_pool, "ArmHand")
    s1_feats = s1_rank.head(args.topk_s1).index.tolist()
    s2_feats = s2_rank.head(args.topk_s2).index.tolist()
    print(f"  stage1: top {len(s1_feats)} of {len(bvp)+len(eda)} BVP+EDA features")
    print(f"  stage2 ({args.s2_modality}): top {len(s2_feats)} of {len(s2_pool)} features")

    # Multiview: train on each resample table separately
    views = args.views.split(",")
    pain_train_list, arm_train_list = [], []
    pain_val_list, arm_val_list = [], []
    ref_train, ref_val = None, None
    print(f"\n>>> Multiview training on {len(views)} tables...")
    for view in views:
        print(f"  view: {view}")
        df_all, feat_cols = load_clean_features(RESAMPLE_PATHS[view])
        s1_feats_v = [f for f in s1_feats if f in feat_cols]
        s2_feats_v = [f for f in s2_feats if f in feat_cols]
        train_s1, val_s1, train_pain, val_pain, s1_spec, s1_cal = fit_s1_one_view(df_all, feat_cols, s1_feats_v)
        train_s2, val_s2, train_arm, val_arm, s2_spec, s2_cal = fit_s2_one_view(df_all, feat_cols, s2_feats_v)
        if ref_train is None:
            ref_train, ref_val = train_s1, val_s1
            pain_train_list.append(train_pain)
            arm_train_list.append(align(train_s2, train_arm, ref_train))
            pain_val_list.append(val_pain)
            arm_val_list.append(align(val_s2, val_arm, ref_val))
        else:
            pain_train_list.append(align(train_s1, train_pain, ref_train))
            arm_train_list.append(align(train_s2, train_arm, ref_train))
            pain_val_list.append(align(val_s1, val_pain, ref_val))
            arm_val_list.append(align(val_s2, val_arm, ref_val))

    pain_t = np.mean(pain_train_list, axis=0)
    pain_v = np.mean(pain_val_list, axis=0)
    arm_t = np.mean(arm_train_list, axis=0)
    arm_v = np.mean(arm_val_list, axis=0)

    # Decode + metrics
    y_true_v = class_codes_3(ref_val["class"])
    y_true_t = class_codes_3(ref_train["class"])
    y_pred_t = decode_subjectwise(ref_train, pain_t, arm_t, w=tuple(args.w))
    y_pred_v = decode_subjectwise(ref_val, pain_v, arm_v, w=tuple(args.w))

    ps_loso = per_subject(ref_train, y_true_t, y_pred_t)
    ps_val = per_subject(ref_val, y_true_v, y_pred_v)
    stats_loso = stats(ps_loso)
    stats_val = stats(ps_val)

    # Save
    ps_loso.to_csv(out_dir / "loso_per_subject.csv", index=False)
    ps_val.to_csv(out_dir / "val_per_subject.csv", index=False)
    preds_out = ref_val[["subject", "segment_id", "class"]].copy()
    preds_out["true_y"] = y_true_v
    preds_out["pred_y"] = y_pred_v
    preds_out["stage1_pain_prob"] = pain_v
    preds_out["stage2_arm_prob"] = arm_v
    preds_out["pred_class"] = ["NoPain" if i == 0 else ("PainArm" if i == 1 else "PainHand") for i in y_pred_v]
    preds_out.to_parquet(out_dir / "validation_predictions.parquet", index=False)

    cm_v = pd.crosstab(pd.Series(y_true_v, name="true"), pd.Series(y_pred_v, name="pred")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
    plot_confusion(cm_v, ["NoPain", "Arm", "Hand"], f"Final Val ({stats_val['mean']:.4f})", out_dir / "confusion_val.png")
    cm_t = pd.crosstab(pd.Series(y_true_t, name="true"), pd.Series(y_pred_t, name="pred")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
    plot_confusion(cm_t, ["NoPain", "Arm", "Hand"], f"Final LOSO ({stats_loso['mean']:.4f})", out_dir / "confusion_loso.png")
    plot_per_subject(ps_loso, f"LOSO F1 per subject (mean={stats_loso['mean']:.3f} std={stats_loso['std']:.3f})", out_dir / "per_subject_loso.png", value_col="f1")
    plot_per_subject(ps_val, f"VAL F1 per subject (mean={stats_val['mean']:.3f} std={stats_val['std']:.3f})", out_dir / "per_subject_val.png", value_col="f1")

    summary = {
        "views": views,
        "topk_s1": args.topk_s1, "topk_s2": args.topk_s2,
        "decoder_w": list(args.w),
        "loso": stats_loso, "validation": stats_val,
        "s1_features": s1_feats, "s2_features": s2_feats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS")
    print("=" * 60)
    print(f"LOSO:  mean={stats_loso['mean']:.4f}  std={stats_loso['std']:.4f}  min={stats_loso['min']:.3f}  max={stats_loso['max']:.3f}  n<0.5={stats_loso['n_below_05']}")
    print(f"VAL:   mean={stats_val['mean']:.4f}  std={stats_val['std']:.4f}  min={stats_val['min']:.3f}  max={stats_val['max']:.3f}  n<0.5={stats_val['n_below_05']}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--views", default="truncate1022,linear1022,poly1022")
    p.add_argument("--topk-s1", type=int, default=98)
    p.add_argument("--topk-s2", type=int, default=30)
    p.add_argument("--s2-modality", choices=["resp", "eda_resp", "bvp_resp", "all"], default="eda_resp")
    p.add_argument("--w", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    p.add_argument("--output-dir", default="results/final/final_pipeline")
    args = p.parse_args()
    main(args)
