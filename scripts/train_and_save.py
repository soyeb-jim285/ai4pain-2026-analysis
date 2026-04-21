from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.final_pipeline import (
    ARM_HAND, ModelSpec, armhand_binary, build_norm_map, class_codes_3,
    decode_joint_weighted, fit_binary_calibrator, load_clean_features,
    make_binary_model, make_scaler, metrics_multiclass, plot_confusion,
    plot_per_subject, stage1_feature_sets, stage2_feature_sets,
)


def fit_stage1_full(train: pd.DataFrame, feats: list[str], spec: ModelSpec, scaler_name: str, calibration: str) -> dict:
    scaler = make_scaler(scaler_name)
    X = scaler.fit_transform(train[feats].to_numpy(dtype=np.float32))
    y = (train["class"] != "NoPain").astype(int).to_numpy()
    mdl = make_binary_model(spec)
    mdl.fit(X, y)

    # LOSO train probs for calibration
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    for subj in sorted(train["subject"].unique()):
        mask = train["subject"] == subj
        tr = train[~mask].reset_index(drop=True)
        te = train[mask].reset_index(drop=True)
        sc = make_scaler(scaler_name)
        Xt = sc.fit_transform(tr[feats].to_numpy(dtype=np.float32))
        Xe = sc.transform(te[feats].to_numpy(dtype=np.float32))
        yt = (tr["class"] != "NoPain").astype(int).to_numpy()
        m = make_binary_model(spec)
        m.fit(Xt, yt)
        train_probs[mask.to_numpy()] = m.predict_proba(Xe).astype(np.float32)
    cal = fit_binary_calibrator(calibration, train_probs[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    return {"scaler": scaler, "model": mdl, "calibrator": cal, "features": feats, "calibration": calibration}


def fit_stage2_full(train_full: pd.DataFrame, feats: list[str], spec: ModelSpec, scaler_name: str, calibration: str) -> dict:
    train = train_full[train_full["class"].isin(ARM_HAND)].reset_index(drop=True)
    scaler = make_scaler(scaler_name)
    X = scaler.fit_transform(train[feats].to_numpy(dtype=np.float32))
    y = armhand_binary(train["class"])
    mdl = make_binary_model(spec)
    mdl.fit(X, y)

    # LOSO for calibration on pain subset
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    for subj in sorted(train["subject"].unique()):
        mask = train["subject"] == subj
        tr = train[~mask].reset_index(drop=True)
        te = train[mask].reset_index(drop=True)
        sc = make_scaler(scaler_name)
        Xt = sc.fit_transform(tr[feats].to_numpy(dtype=np.float32))
        Xe = sc.transform(te[feats].to_numpy(dtype=np.float32))
        yt = armhand_binary(tr["class"])
        m = make_binary_model(spec)
        m.fit(Xt, yt)
        train_probs[mask.to_numpy()] = m.predict_proba(Xe).astype(np.float32)
    cal = fit_binary_calibrator(calibration, train_probs[:, 0], (train["class"] == "PainArm").astype(int).to_numpy())
    return {"scaler": scaler, "model": mdl, "calibrator": cal, "features": feats, "calibration": calibration}


def predict(df_norm: pd.DataFrame, stage1: dict, stage2: dict, w0: float, w1: float, w2: float) -> np.ndarray:
    X1 = stage1["scaler"].transform(df_norm[stage1["features"]].to_numpy(dtype=np.float32))
    p1 = stage1["model"].predict_proba(X1).astype(np.float32)
    nop_cal = stage1["calibrator"](p1[:, 0])
    pain_cal = 1.0 - nop_cal
    s1_probs = np.column_stack([nop_cal, pain_cal]).astype(np.float32)

    X2 = stage2["scaler"].transform(df_norm[stage2["features"]].to_numpy(dtype=np.float32))
    p2 = stage2["model"].predict_proba(X2).astype(np.float32)
    arm_cal = stage2["calibrator"](p2[:, 0])
    s2_probs = np.column_stack([arm_cal, 1 - arm_cal]).astype(np.float32)

    y_pred = np.zeros(len(df_norm), dtype=int)
    for subj in sorted(df_norm["subject"].unique()):
        mask = (df_norm["subject"] == subj).to_numpy()
        y_pred[mask] = decode_joint_weighted(s1_probs[mask], s2_probs[mask], w0=w0, w1=w1, w2=w2)
    return y_pred, s1_probs, s2_probs


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features(args.feature_parquet)
    norm_map = build_norm_map(df_all, feat_cols)

    s1_df = norm_map[args.stage1_norm]
    s2_df = norm_map[args.stage2_norm]
    train_s1 = s1_df[s1_df["split"] == "train"].reset_index(drop=True)
    val_s1 = s1_df[s1_df["split"] == "validation"].reset_index(drop=True)
    train_s2 = s2_df[s2_df["split"] == "train"].reset_index(drop=True)
    val_s2 = s2_df[s2_df["split"] == "validation"].reset_index(drop=True)

    s1_feats = stage1_feature_sets(train_s1, feat_cols)[args.stage1_feature_set]
    s2_feats = stage2_feature_sets(train_s2, feat_cols)[args.stage2_feature_set]

    s1_spec = ModelSpec(args.stage1_model, {"n_estimators": args.stage1_xgb_n_estimators, "max_depth": args.stage1_xgb_max_depth, "learning_rate": args.stage1_xgb_learning_rate, "subsample": 1.0, "colsample_bytree": 1.0, "C": args.stage1_logreg_c, "gamma": "scale"})
    s2_spec = ModelSpec(args.stage2_model, {"n_estimators": args.stage2_xgb_n_estimators, "max_depth": args.stage2_xgb_max_depth, "learning_rate": args.stage2_xgb_learning_rate, "subsample": 1.0, "colsample_bytree": 1.0, "C": args.stage2_logreg_c, "gamma": "scale"})

    print("Fitting stage 1...")
    s1 = fit_stage1_full(train_s1, s1_feats, s1_spec, args.stage1_scaler, args.stage1_calibration)
    print("Fitting stage 2...")
    s2 = fit_stage2_full(train_s2, s2_feats, s2_spec, args.stage2_scaler, args.stage2_calibration)

    # Predict validation
    y_pred, s1_probs, s2_probs = predict(val_s1, s1, s2, args.w0, args.w1, args.w2)
    y_true = class_codes_3(val_s1["class"])
    m = metrics_multiclass(y_true, y_pred)
    print(f"Validation: {m}")

    # Save models
    joblib.dump(s1, out_dir / "stage1.joblib")
    joblib.dump(s2, out_dir / "stage2.joblib")

    # Save config + normalizer stats
    norm_stats = {
        "stage1_norm": args.stage1_norm, "stage2_norm": args.stage2_norm,
        "stage1_feature_set": args.stage1_feature_set, "stage2_feature_set": args.stage2_feature_set,
        "w0": args.w0, "w1": args.w1, "w2": args.w2,
        "feature_parquet": str(args.feature_parquet),
        "feat_cols": feat_cols,
    }
    (out_dir / "config.json").write_text(json.dumps(norm_stats, indent=2))

    # Save validation predictions + confusion
    val_out = val_s1[["subject", "segment_id", "class"]].copy()
    val_out["true_y"] = y_true
    val_out["pred_y"] = y_pred
    val_out["pred_class"] = ["NoPain" if i == 0 else ("PainArm" if i == 1 else "PainHand") for i in y_pred]
    val_out["stage1_nopain_prob"] = s1_probs[:, 0]
    val_out["stage2_arm_prob"] = s2_probs[:, 0]
    val_out.to_parquet(out_dir / "validation_predictions.parquet", index=False)
    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
    plot_confusion(cm, ["NoPain", "Arm", "Hand"], f"Validation ({m['macro_f1']:.4f})", out_dir / "confusion.png")
    pd.DataFrame([m]).to_csv(out_dir / "summary.csv", index=False)

    print(f"\nSaved models + config to: {out_dir}")
    print(f"  stage1.joblib, stage2.joblib, config.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature-parquet", default="results/tables/all_features_merged_1022.parquet")
    p.add_argument("--stage1-norm", default="subject_robust")
    p.add_argument("--stage1-feature-set", default="bvp_eda_core")
    p.add_argument("--stage1-model", default="xgb")
    p.add_argument("--stage1-scaler", default="std")
    p.add_argument("--stage1-calibration", default="sigmoid")
    p.add_argument("--stage1-xgb-n-estimators", type=int, default=200)
    p.add_argument("--stage1-xgb-max-depth", type=int, default=4)
    p.add_argument("--stage1-xgb-learning-rate", type=float, default=0.08)
    p.add_argument("--stage1-logreg-c", type=float, default=1.0)
    p.add_argument("--stage2-norm", default="subject_robust")
    p.add_argument("--stage2-feature-set", default="resp_all")
    p.add_argument("--stage2-model", default="xgb")
    p.add_argument("--stage2-scaler", default="robust")
    p.add_argument("--stage2-calibration", default="isotonic")
    p.add_argument("--stage2-xgb-n-estimators", type=int, default=200)
    p.add_argument("--stage2-xgb-max-depth", type=int, default=4)
    p.add_argument("--stage2-xgb-learning-rate", type=float, default=0.08)
    p.add_argument("--stage2-logreg-c", type=float, default=1.0)
    p.add_argument("--w0", type=float, default=0.8)
    p.add_argument("--w1", type=float, default=1.2)
    p.add_argument("--w2", type=float, default=1.4)
    p.add_argument("--output-dir", default="results/final/saved_model")
    args = p.parse_args()
    main(args)
