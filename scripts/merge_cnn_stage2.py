"""Merge Kaggle CNN stage 2 probs with classical stage 1.

Expects Kaggle output in: results/final/stage2_cnn_kaggle/
  - loso_arm_probs.parquet  (LOSO probs for train)
  - val_arm_probs.parquet   (validation probs)

Run:
  uv run python scripts/merge_cnn_stage2.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, channel_of, class_codes_3,
    decode_joint_weighted, fit_binary_calibrator, fit_binary_proba,
    load_clean_features, metrics_multiclass, plot_confusion, plot_per_subject,
    unique_by_canonical,
)

CNN_DIR = Path("results/final/stage2_cnn_kaggle")
OUT = Path("results/final/combined_cnn")
OUT.mkdir(parents=True, exist_ok=True)


def fit_stage1(df_all, feat_cols):
    s1_feats = unique_by_canonical([c for c in feat_cols if channel_of(c) in ("bvp", "eda")])
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


def decode(df, pp, ap, w=(1.0, 1.0, 1.0)):
    s1 = np.column_stack([1 - pp, pp]).astype(np.float32)
    s2 = np.column_stack([ap, 1 - ap]).astype(np.float32)
    y = np.zeros(len(df), dtype=int)
    for s in sorted(df["subject"].unique()):
        m = (df["subject"] == s).to_numpy()
        y[m] = decode_joint_weighted(s1[m], s2[m], *w)
    return y


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
    if not (CNN_DIR / "val_arm_probs.parquet").exists():
        print(f"ERROR: CNN outputs not found in {CNN_DIR}")
        print("Steps:")
        print("  1. uv run jupytext --to ipynb kaggle/stage2_cnn.py -o kaggle/stage2_cnn.ipynb")
        print("  2. bash kaggle/upload_stage2_cache.sh   # push cache to dataset")
        print("  3. cp kaggle/kernel-metadata-stage2.json kaggle/kernel-metadata.json")
        print("  4. cd kaggle && kaggle kernels push")
        print("  5. wait for Kaggle to finish, then:")
        print("     kaggle kernels output soyebjim/ai4pain-2026-stage-2-cnn-arm-vs-hand -p results/final/stage2_cnn_kaggle")
        sys.exit(1)

    cnn_loso = pd.read_parquet(CNN_DIR / "loso_arm_probs.parquet")
    cnn_val = pd.read_parquet(CNN_DIR / "val_arm_probs.parquet")
    print(f"CNN LOSO probs: {len(cnn_loso)} rows")
    print(f"CNN VAL probs:  {len(cnn_val)} rows")

    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    train_s1, val_s1, pain_t, pain_v = fit_stage1(df_all, feat_cols)

    # Fit isotonic calibration of CNN LOSO probs vs true arm label
    y_arm_loso = (cnn_loso["class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator("isotonic", cnn_loso["arm_prob_cnn"].to_numpy(), y_arm_loso)

    # Expand to all segments (NoPain -> 0.5)
    def expand(full_meta, pain_df, probs):
        mp = dict(zip(pain_df["segment_id"], probs))
        return np.array([mp.get(sid, 0.5) for sid in full_meta["segment_id"]], dtype=np.float32)

    arm_t = expand(train_s1, cnn_loso, cal(cnn_loso["arm_prob_cnn"].to_numpy()).astype(np.float32))
    arm_v = expand(val_s1, cnn_val, cal(cnn_val["arm_prob_cnn"].to_numpy()).astype(np.float32))

    ytt = class_codes_3(train_s1["class"])
    ytv = class_codes_3(val_s1["class"])
    ypt = decode(train_s1, pain_t, arm_t)
    ypv = decode(val_s1, pain_v, arm_v)

    loso = stats_ps(train_s1, ytt, ypt)
    val_ = stats_ps(val_s1, ytv, ypv)

    print(f"\n=== COMBINED (Stage1 XGB + Stage2 CNN) ===")
    print(f"LOSO: mean={loso['mean']:.4f} std={loso['std']:.4f} min={loso['min']:.3f} n<0.5={loso['n_below_05']}")
    print(f"VAL:  mean={val_['mean']:.4f} std={val_['std']:.4f} min={val_['min']:.3f} n<0.5={val_['n_below_05']}")

    # Save
    val_out = val_s1[["subject", "segment_id", "class"]].copy()
    val_out["true_y"] = ytv
    val_out["pred_y"] = ypv
    val_out["stage1_pain_prob"] = pain_v
    val_out["stage2_arm_prob"] = arm_v
    val_out["pred_class"] = ["NoPain" if i == 0 else ("PainArm" if i == 1 else "PainHand") for i in ypv]
    val_out.to_parquet(OUT / "validation_predictions.parquet", index=False)
    cm = pd.crosstab(pd.Series(ytv, name="true"), pd.Series(ypv, name="pred")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
    plot_confusion(cm, ["NoPain", "Arm", "Hand"], f"Combined CNN ({val_['mean']:.4f})", OUT / "confusion.png")
    pd.DataFrame([{"split": "loso", **loso}, {"split": "val", **val_}]).to_csv(OUT / "summary.csv", index=False)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
