from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND,
    HAS_XGB,
    ModelSpec,
    armhand_binary,
    build_norm_map,
    channel_of,
    class_codes_3,
    exact_count_decode,
    fit_binary_calibrator,
    fit_binary_proba,
    load_clean_features,
    metrics_binary,
    plot_calibration,
    plot_confusion,
    plot_per_subject,
    stage2_feature_sets,
)


def default_feature_parquet(resample_tag: str) -> Path:
    return Path(f"results/tables/all_features_merged_{resample_tag}.parquet")


def build_model_spec(args: argparse.Namespace) -> ModelSpec:
    if args.model == "xgb":
        return ModelSpec("xgb", {
            "n_estimators": args.xgb_n_estimators,
            "max_depth": args.xgb_max_depth,
            "learning_rate": args.xgb_learning_rate,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree,
        })
    if args.model == "rf":
        return ModelSpec("rf", {"n_estimators": args.rf_n_estimators, "max_depth": args.rf_max_depth})
    if args.model == "logreg":
        return ModelSpec("logreg", {"C": args.logreg_c})
    if args.model == "svm_linear":
        return ModelSpec("svm_linear", {"C": args.svm_c})
    if args.model == "svm_rbf":
        return ModelSpec("svm_rbf", {"C": args.svm_c, "gamma": args.svm_gamma})
    raise ValueError(args.model)


def exact12_arm_predictions(df_sub: pd.DataFrame, arm_scores: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-arm_scores[mask])]
        arm_idx = set(order[:12].tolist())
        pred[mask] = [0 if i in arm_idx else 1 for i in idx]
    return pred


def adapt_with_true_nopain(df_norm: pd.DataFrame, df_global: pd.DataFrame, feat_cols: list[str], mode: str) -> pd.DataFrame:
    out = df_norm.copy()
    for subject in sorted(out["subject"].unique()):
        mask = out["subject"] == subject
        gsub = df_global[df_global["subject"] == subject].reset_index(drop=True)
        base = gsub["class"] == "NoPain"
        if int(base.sum()) == 0:
            continue
        mu = gsub.loc[base, feat_cols].mean(axis=0)
        if mode == "center":
            out.loc[mask, feat_cols] = (out.loc[mask, feat_cols] - mu).astype(np.float32)
        elif mode == "z":
            sd = gsub.loc[base, feat_cols].std(axis=0, ddof=0)
            sd = sd.where(sd > 0, 1.0)
            out.loc[mask, feat_cols] = ((out.loc[mask, feat_cols] - mu) / sd).astype(np.float32)
        else:
            raise ValueError(mode)
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out


def main(args: argparse.Namespace) -> None:
    feature_fp = args.feature_parquet or default_feature_parquet(args.resample_tag)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features(feature_fp)
    norm_map = build_norm_map(df_all, feat_cols)
    df_norm = norm_map[args.norm]
    if args.anchor_mode != "none":
        df_norm = adapt_with_true_nopain(df_norm, df_all, feat_cols, mode=args.anchor_mode)
    train_df = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val_df = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feature_sets = stage2_feature_sets(train_df, feat_cols)
    selected = feature_sets[args.feature_set]
    spec = build_model_spec(args)

    train_pain = train_df[train_df["class"].isin(ARM_HAND)].reset_index(drop=True)
    val_pain = val_df[val_df["class"].isin(ARM_HAND)].reset_index(drop=True)
    train_probs = np.zeros((len(train_pain), 2), dtype=np.float32)
    for subject in sorted(train_pain["subject"].unique()):
        tr = train_pain[train_pain["subject"] != subject].reset_index(drop=True)
        te = train_pain[train_pain["subject"] == subject].reset_index(drop=True)
        probs = fit_binary_proba(tr, te, selected, armhand_binary(tr["class"]), scaler_name=args.scaler, spec=spec)
        train_probs[train_pain["subject"] == subject] = probs
    val_probs = fit_binary_proba(train_pain, val_pain, selected, armhand_binary(train_pain["class"]), scaler_name=args.scaler, spec=spec)

    cal = fit_binary_calibrator(args.calibration, train_probs[:, 0], (train_pain["class"] == "PainArm").astype(int).to_numpy())
    train_arm = cal(train_probs[:, 0])
    val_arm = cal(val_probs[:, 0])

    rows = []
    subject_rows = []
    pred_rows = []
    for split_name, df_sub, arm_scores in (("train_loso", train_pain, train_arm), ("validation", val_pain, val_arm)):
        y_true = (df_sub["class"] == "PainHand").astype(int).to_numpy()
        y_pred = exact12_arm_predictions(df_sub, arm_scores)
        met = metrics_binary(y_true, y_pred)
        rows.append({"split": split_name, **met})
        for subject in sorted(df_sub["subject"].unique()):
            mask = (df_sub["subject"] == subject).to_numpy()
            m = metrics_binary(y_true[mask], y_pred[mask])
            subject_rows.append({"split": split_name, "subject": int(subject), **m})
        tmp = df_sub[["subject", "segment_id", "class"]].copy().reset_index(drop=True)
        tmp["true_y"] = y_true
        tmp["pred_y"] = y_pred
        tmp["arm_prob"] = arm_scores
        tmp["split"] = split_name
        pred_rows.append(tmp)

    summary = pd.DataFrame(rows)
    per_subject = pd.DataFrame(subject_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_subject.to_csv(out_dir / "per_subject.csv", index=False)
    preds.to_parquet(out_dir / "predictions.parquet", index=False)

    val_pred = preds[preds["split"] == "validation"].copy()
    cm = pd.crosstab(val_pred["true_y"], val_pred["pred_y"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
    plot_confusion(cm, ["Arm", "Hand"], "Stage 2 Validation Confusion", out_dir / "confusion_validation.png")
    plot_per_subject(per_subject[per_subject["split"] == "validation"], "Stage 2 Validation Per-Subject Macro-F1", out_dir / "per_subject_validation.png")
    plot_calibration((val_pred["class"] == "PainArm").astype(int).to_numpy(), np.clip(val_pred["arm_prob"].to_numpy(), 1e-6, 1 - 1e-6), "Stage 2 Calibration", out_dir / "calibration_validation.png")

    report = [
        "# Stage 2 Run",
        f"- feature parquet: `{feature_fp}`",
        f"- normalization: `{args.norm}`",
        f"- anchor mode: `{args.anchor_mode}`",
        f"- feature set: `{args.feature_set}` ({len(selected)} features)",
        f"- model: `{spec.name}` {spec.params}",
        f"- scaler: `{args.scaler}`",
        f"- calibration: `{args.calibration}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
    ]
    (out_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature-parquet", type=Path, default=None)
    p.add_argument("--resample-tag", default="1022")
    p.add_argument("--norm", choices=["global", "subject_z", "subject_robust"], default="subject_z")
    p.add_argument("--anchor-mode", choices=["none", "center", "z"], default="none")
    p.add_argument("--feature-set", choices=["resp_all", "resp_top20", "resp_bvp5"], default="resp_all")
    p.add_argument("--model", choices=["xgb", "rf", "logreg", "svm_linear", "svm_rbf"], default="logreg")
    p.add_argument("--scaler", choices=["std", "robust"], default="robust")
    p.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], default="isotonic")
    p.add_argument("--xgb-n-estimators", type=int, default=200)
    p.add_argument("--xgb-max-depth", type=int, default=4)
    p.add_argument("--xgb-learning-rate", type=float, default=0.08)
    p.add_argument("--xgb-subsample", type=float, default=1.0)
    p.add_argument("--xgb-colsample-bytree", type=float, default=1.0)
    p.add_argument("--rf-n-estimators", type=int, default=400)
    p.add_argument("--rf-max-depth", default=None)
    p.add_argument("--logreg-c", type=float, default=1.0)
    p.add_argument("--svm-c", type=float, default=3.0)
    p.add_argument("--svm-gamma", default="scale")
    p.add_argument("--output-dir", default="results/final/stage2")
    args = p.parse_args()
    if args.model == "xgb" and not HAS_XGB:
        raise SystemExit("xgboost is not installed")
    main(args)
