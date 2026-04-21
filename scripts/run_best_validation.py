from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ModelSpec, build_norm_map, class_codes_3, decode_joint_weighted,
    fit_binary_calibrator, fit_binary_proba, load_clean_features,
    metrics_multiclass, plot_confusion, plot_per_subject, stage1_feature_sets,
)


def main() -> None:
    out_dir = Path("results/final/best_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    norm_map = build_norm_map(df_all, feat_cols)
    df_norm = norm_map["subject_robust"]
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feats = stage1_feature_sets(train, feat_cols)["bvp_eda_core"]
    spec = ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})

    # Stage 1 LOSO calibration
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    for subj in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subj].reset_index(drop=True)
        te = train[train["subject"] == subj].reset_index(drop=True)
        p = fit_binary_proba(tr, te, feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
        train_probs[train["subject"] == subj] = p
    val_probs = fit_binary_proba(train, val, feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    cal = fit_binary_calibrator("sigmoid", train_probs[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    val_nop = cal(val_probs[:, 0])
    val_pain = 1.0 - val_nop

    # Stage 2 from cached suite 31
    cache = pd.read_parquet("results/tables/suite31_validation_predictions.parquet")
    cache = cache[cache["config_id"] == "truncate1022|xgb_subject_z|resp_all_robust|joint_weighted"]
    val["pain_prob"] = val_pain
    merged = val[["subject", "segment_id", "class", "pain_prob"]].merge(
        cache[["segment_id", "stage2_arm_prob"]], on="segment_id",
    ).sort_values(["subject", "segment_id"]).reset_index(drop=True)

    p_pain = merged["pain_prob"].to_numpy()
    p_arm = merged["stage2_arm_prob"].to_numpy()
    s1 = np.column_stack([1 - p_pain, p_pain])
    s2 = np.column_stack([p_arm, 1 - p_arm])
    y_true = class_codes_3(merged["class"])
    y_pred = np.zeros(len(merged), dtype=int)
    for subj in sorted(merged["subject"].unique()):
        mask = (merged["subject"] == subj).to_numpy()
        y_pred[mask] = decode_joint_weighted(s1[mask], s2[mask], w0=0.8, w1=1.2, w2=1.4)
    m = metrics_multiclass(y_true, y_pred)
    print("BEST VALIDATION:", m)

    merged["true_y"] = y_true
    merged["pred_y"] = y_pred
    merged["pred_class"] = ["NoPain" if i == 0 else ("PainArm" if i == 1 else "PainHand") for i in y_pred]
    merged.to_parquet(out_dir / "predictions.parquet", index=False)
    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
    plot_confusion(cm, ["NoPain", "Arm", "Hand"], f"Best Validation ({m['macro_f1']:.4f})", out_dir / "confusion.png")


if __name__ == "__main__":
    main()
