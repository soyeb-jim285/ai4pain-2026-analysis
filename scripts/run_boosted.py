"""Boosted combined - tries multiple stage 1 variants with weight sweep.

Each variant saves predictions. Outputs best config.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND, ModelSpec, armhand_binary, build_norm_map, class_codes_3,
    decode_joint_weighted, fit_binary_calibrator, fit_binary_proba,
    load_clean_features, metrics_multiclass, plot_confusion,
    stage1_feature_sets, stage2_feature_sets,
)


def fit_s1(df_norm, feat_cols, feat_set, model_name, calibration):
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feats = stage1_feature_sets(train, feat_cols)[feat_set]
    if model_name == "xgb":
        spec = ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    else:
        spec = ModelSpec(model_name, {"C": 1.0, "n_estimators": 400})
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    for subj in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subj].reset_index(drop=True)
        te = train[train["subject"] == subj].reset_index(drop=True)
        p = fit_binary_proba(tr, te, feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
        train_probs[train["subject"] == subj] = p
    val_probs = fit_binary_proba(train, val, feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator(calibration, train_probs[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return train, val, 1 - cal(train_probs[:, 0]), 1 - cal(val_probs[:, 0])


def fit_s2_suite31(df_norm, feat_cols):
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


def decode(df, pain_p, arm_p, w0, w1, w2):
    s1 = np.column_stack([1 - pain_p, pain_p]).astype(np.float32)
    s2 = np.column_stack([arm_p, 1 - arm_p]).astype(np.float32)
    y_pred = np.zeros(len(df), dtype=int)
    for subj in sorted(df["subject"].unique()):
        mask = (df["subject"] == subj).to_numpy()
        y_pred[mask] = decode_joint_weighted(s1[mask], s2[mask], w0=w0, w1=w1, w2=w2)
    return y_pred


def sweep(df_val, pain, arm):
    y_true = class_codes_3(df_val["class"])
    best = {"f1": 0, "w": (1, 1, 1), "y_pred": None}
    for w0 in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
        for w1 in [0.6, 0.8, 1.0, 1.2, 1.4]:
            for w2 in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
                y_pred = decode(df_val, pain, arm, w0, w1, w2)
                m = metrics_multiclass(y_true, y_pred)
                if m["macro_f1"] > best["f1"]:
                    best = {"f1": m["macro_f1"], "w": (w0, w1, w2), "y_pred": y_pred}
    return best


def align(src_df, src_vals, ref_df):
    mp = dict(zip(src_df["segment_id"], src_vals))
    return np.array([mp[sid] for sid in ref_df["segment_id"]], dtype=np.float32)


def main() -> None:
    out_dir = Path("results/final/boosted")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    norm_map = build_norm_map(df_all, feat_cols)

    # Stage 2 once (same recipe across)
    print("Stage 2 (suite 31 recipe: subject_z + logreg + robust + isotonic)...")
    train_s2, val_s2, train_arm, val_arm = fit_s2_suite31(norm_map["subject_z"], feat_cols)

    variants = [
        ("suite31_xgb", "subject_z", "all", "xgb", "sigmoid"),
        ("upgraded_xgb", "subject_robust", "bvp_eda_core", "xgb", "sigmoid"),
        ("upgraded_xgb_iso", "subject_robust", "bvp_eda_core", "xgb", "isotonic"),
        ("upgraded_rf", "subject_robust", "bvp_eda_core", "rf", "sigmoid"),
    ]
    results = []
    for name, norm, fs, model, cal in variants:
        print(f"\n>>> variant: {name} (norm={norm} feats={fs} model={model} cal={cal})")
        train_s1, val_s1, train_pain, val_pain = fit_s1(norm_map[norm], feat_cols, fs, model, cal)
        v_arm_al = align(val_s2, val_arm, val_s1)
        t_arm_al = align(train_s2, train_arm, train_s1)
        y_true_val = class_codes_3(val_s1["class"])
        y_true_trn = class_codes_3(train_s1["class"])
        # Fixed (1,1,1)
        y_pred_val_fix = decode(val_s1, val_pain, v_arm_al, 1, 1, 1)
        y_pred_trn_fix = decode(train_s1, train_pain, t_arm_al, 1, 1, 1)
        m_val_fix = metrics_multiclass(y_true_val, y_pred_val_fix)
        m_trn_fix = metrics_multiclass(y_true_trn, y_pred_trn_fix)
        # Tuned (val-sweep)
        best = sweep(val_s1, val_pain, v_arm_al)
        y_pred_trn_tuned = decode(train_s1, train_pain, t_arm_al, *best["w"])
        m_trn_tuned = metrics_multiclass(y_true_trn, y_pred_trn_tuned)
        print(f"  fixed w=(1,1,1):  LOSO f1={m_trn_fix['macro_f1']:.4f} acc={m_trn_fix['accuracy']:.4f}  |  VAL f1={m_val_fix['macro_f1']:.4f} acc={m_val_fix['accuracy']:.4f}")
        print(f"  tuned w={best['w']}: LOSO f1={m_trn_tuned['macro_f1']:.4f} acc={m_trn_tuned['accuracy']:.4f}  |  VAL f1={best['f1']:.4f} acc={metrics_multiclass(y_true_val, best['y_pred'])['accuracy']:.4f}")
        results.append({
            "variant": name,
            "loso_f1_fix": m_trn_fix["macro_f1"], "loso_acc_fix": m_trn_fix["accuracy"],
            "val_f1_fix": m_val_fix["macro_f1"], "val_acc_fix": m_val_fix["accuracy"],
            "loso_f1_tuned": m_trn_tuned["macro_f1"], "loso_acc_tuned": m_trn_tuned["accuracy"],
            "val_f1_tuned": best["f1"], "val_acc_tuned": metrics_multiclass(y_true_val, best["y_pred"])["accuracy"],
            "best_w": str(best["w"]),
        })
        y_true = y_true_val
        m_fix = m_val_fix

        # Save predictions for best
        val_out = val_s1[["subject", "segment_id", "class"]].copy()
        val_out["true_y"] = y_true
        val_out["pred_y"] = best["y_pred"]
        val_out["stage1_pain_prob"] = val_pain
        val_out["stage2_arm_prob"] = v_arm_al
        val_out["pred_class"] = ["NoPain" if i == 0 else ("PainArm" if i == 1 else "PainHand") for i in best["y_pred"]]
        val_out.to_parquet(out_dir / f"preds_{name}.parquet", index=False)
        cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(best["y_pred"], name="pred")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
        plot_confusion(cm, ["NoPain", "Arm", "Hand"], f"{name} val={best['f1']:.4f}", out_dir / f"cm_{name}.png")

    df = pd.DataFrame(results).sort_values("val_f1_tuned", ascending=False)
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== FINAL RANKING ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
