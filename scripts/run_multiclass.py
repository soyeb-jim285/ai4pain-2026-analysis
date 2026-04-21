from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    CLASS_ORDER_3,
    HAS_XGB,
    ModelSpec,
    build_norm_map,
    class_codes_3,
    exact_count_decode,
    fit_multiclass_proba,
    load_clean_features,
    metrics_multiclass,
    multiclass_feature_sets,
    plot_calibration,
    plot_confusion,
    plot_per_subject,
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


def fit_loso_and_validation(df_norm: pd.DataFrame, feat_cols: list[str], spec: ModelSpec, scaler_name: str):
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    train_probs = np.zeros((len(train), len(CLASS_ORDER_3)), dtype=np.float32)
    for subject in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subject].reset_index(drop=True)
        te = train[train["subject"] == subject].reset_index(drop=True)
        probs = fit_multiclass_proba(tr, te, feat_cols, class_codes_3(tr["class"]), scaler_name=scaler_name, spec=spec, n_classes=len(CLASS_ORDER_3))
        train_probs[train["subject"] == subject] = probs
    val_probs = fit_multiclass_proba(train, val, feat_cols, class_codes_3(train["class"]), scaler_name=scaler_name, spec=spec, n_classes=len(CLASS_ORDER_3))
    return train, val, train_probs, val_probs


def decode_predictions(df_sub: pd.DataFrame, probs: np.ndarray, decoder: str) -> np.ndarray:
    if decoder == "argmax":
        return np.argmax(probs, axis=1)
    y_pred = np.zeros(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        subject_probs = np.clip(probs[mask], 1e-6, 1.0 - 1e-6)
        y_pred[mask] = exact_count_decode(np.log(subject_probs), [12, 12, 12])
    return y_pred


def main(args: argparse.Namespace) -> None:
    feature_fp = args.feature_parquet or default_feature_parquet(args.resample_tag)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features(feature_fp)
    norm_map = build_norm_map(df_all, feat_cols)
    df_norm = norm_map[args.norm]
    train_df = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    feature_sets = multiclass_feature_sets(train_df, feat_cols)
    selected = feature_sets[args.feature_set]
    spec = build_model_spec(args)

    train_df, val_df, train_probs, val_probs = fit_loso_and_validation(df_norm, selected, spec, scaler_name=args.scaler)

    rows = []
    subject_rows = []
    pred_rows = []
    for split_name, df_sub, probs in (("train_loso", train_df, train_probs), ("validation", val_df, val_probs)):
        y_true = class_codes_3(df_sub["class"])
        y_pred = decode_predictions(df_sub, probs, decoder=args.decoder)
        met = metrics_multiclass(y_true, y_pred)
        rows.append({"split": split_name, **met})
        for subject in sorted(df_sub["subject"].unique()):
            mask = (df_sub["subject"] == subject).to_numpy()
            m = metrics_multiclass(y_true[mask], y_pred[mask])
            subject_rows.append({"split": split_name, "subject": int(subject), **m})
        tmp = df_sub[["subject", "segment_id", "class"]].copy().reset_index(drop=True)
        tmp["true_y"] = y_true
        tmp["pred_y"] = y_pred
        tmp["pred_class"] = [CLASS_ORDER_3[i] for i in y_pred]
        tmp["prob_nopain"] = probs[:, 0]
        tmp["prob_painarm"] = probs[:, 1]
        tmp["prob_painhand"] = probs[:, 2]
        tmp["split"] = split_name
        pred_rows.append(tmp)

    summary = pd.DataFrame(rows)
    per_subject = pd.DataFrame(subject_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_subject.to_csv(out_dir / "per_subject.csv", index=False)
    preds.to_parquet(out_dir / "predictions.parquet", index=False)

    val_pred = preds[preds["split"] == "validation"].copy()
    cm = pd.crosstab(val_pred["true_y"], val_pred["pred_y"]).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
    plot_confusion(cm, CLASS_ORDER_3, "Direct Multiclass Validation Confusion", out_dir / "confusion_validation.png")
    plot_per_subject(per_subject[per_subject["split"] == "validation"], "Direct Multiclass Validation Per-Subject Macro-F1", out_dir / "per_subject_validation.png")
    for cls_idx, cls_name, prob_col in ((0, "NoPain", "prob_nopain"), (1, "PainArm", "prob_painarm"), (2, "PainHand", "prob_painhand")):
        plot_calibration((val_pred["true_y"] == cls_idx).astype(int).to_numpy(), np.clip(val_pred[prob_col].to_numpy(), 1e-6, 1 - 1e-6), f"{cls_name} Calibration", out_dir / f"calibration_{cls_name.lower()}.png")

    report = [
        "# Direct Multiclass Run",
        f"- feature parquet: `{feature_fp}`",
        f"- normalization: `{args.norm}`",
        f"- feature set: `{args.feature_set}` ({len(selected)} features)",
        f"- model: `{spec.name}` {spec.params}",
        f"- scaler: `{args.scaler}`",
        f"- decoder: `{args.decoder}`",
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
    p.add_argument("--norm", choices=["global", "subject_z", "subject_robust"], default="subject_robust")
    p.add_argument("--feature-set", choices=["bvp_eda_resp", "top80", "all_top120", "all_modalities"], default="top80")
    p.add_argument("--model", choices=["xgb", "rf", "logreg", "svm_linear", "svm_rbf"], default="xgb")
    p.add_argument("--scaler", choices=["std", "robust"], default="std")
    p.add_argument("--decoder", choices=["exact_counts", "argmax"], default="exact_counts")
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
    p.add_argument("--output-dir", default="results/final/multiclass")
    args = p.parse_args()
    if args.model == "xgb" and not HAS_XGB:
        raise SystemExit("xgboost is not installed")
    main(args)
