from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    HAS_XGB,
    META_COLS,
    ModelSpec,
    apply_subject_robust,
    build_norm_map,
    class_codes_3,
    ensure_parent,
    exact12_binary_predictions,
    fit_binary_calibrator,
    fit_binary_proba,
    load_clean_features,
    make_binary_model,
    make_scaler,
    metrics_binary,
    pain_binary,
    plot_calibration,
    plot_confusion,
    plot_per_subject,
    stage1_anchor_scores,
    stage1_feature_sets,
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
    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    for subject in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subject].reset_index(drop=True)
        te = train[train["subject"] == subject].reset_index(drop=True)
        probs = fit_binary_proba(tr, te, feat_cols, pain_binary(tr["class"]), scaler_name=scaler_name, spec=spec)
        train_probs[train["subject"] == subject] = probs
    val_probs = fit_binary_proba(train, val, feat_cols, pain_binary(train["class"]), scaler_name=scaler_name, spec=spec)
    return train, val, train_probs, val_probs


def maybe_xgb_curve(train_df: pd.DataFrame, val_df: pd.DataFrame, feat_cols: list[str], spec: ModelSpec, scaler_name: str) -> pd.DataFrame:
    if spec.name != "xgb":
        return pd.DataFrame()
    scaler = make_scaler(scaler_name)
    X_tr = scaler.fit_transform(train_df[feat_cols].to_numpy(dtype=np.float32))
    X_va = scaler.transform(val_df[feat_cols].to_numpy(dtype=np.float32))
    y_tr = pain_binary(train_df["class"])
    y_va = pain_binary(val_df["class"])
    mdl = make_binary_model(spec)
    mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)
    ev = mdl.evals_result()
    return pd.DataFrame({
        "iter": np.arange(1, len(ev.get("validation_0", {}).get("logloss", [])) + 1),
        "train_logloss": ev.get("validation_0", {}).get("logloss", []),
        "val_logloss": ev.get("validation_1", {}).get("logloss", []),
    })


def main(args: argparse.Namespace) -> None:
    feature_fp = args.feature_parquet or default_feature_parquet(args.resample_tag)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features(feature_fp)
    norm_map = build_norm_map(df_all, feat_cols)
    df_norm = norm_map[args.norm]
    train_df = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    feature_sets = stage1_feature_sets(train_df, feat_cols)
    selected = feature_sets[args.feature_set]
    spec = build_model_spec(args)

    train_df, val_df, train_probs_raw, val_probs_raw = fit_loso_and_validation(df_norm, selected, spec, scaler_name=args.scaler)
    calibrator = fit_binary_calibrator(args.calibration, train_probs_raw[:, 0], 1 - pain_binary(train_df["class"]))
    train_nop = calibrator(train_probs_raw[:, 0])
    val_nop = calibrator(val_probs_raw[:, 0])
    train_final = train_nop.copy()
    val_final = val_nop.copy()
    if args.anchor_mode != "none":
        global_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
        global_val = df_all[df_all["split"] == "validation"].reset_index(drop=True)
        train_final = stage1_anchor_scores(train_df, global_train, selected, train_nop, mode=args.anchor_mode, lam=args.anchor_lambda)
        val_final = stage1_anchor_scores(val_df, global_val, selected, val_nop, mode=args.anchor_mode, lam=args.anchor_lambda)

    rows = []
    pred_rows = []
    subject_rows = []
    for split_name, df_sub, base_scores, final_scores in (("train_loso", train_df, train_nop, train_final), ("validation", val_df, val_nop, val_final)):
        y_true = pain_binary(df_sub["class"])
        y_pred = exact12_binary_predictions(df_sub, final_scores)
        met = metrics_binary(y_true, y_pred)
        rows.append({"split": split_name, **met})
        for subject in sorted(df_sub["subject"].unique()):
            mask = (df_sub["subject"] == subject).to_numpy()
            m = metrics_binary(y_true[mask], y_pred[mask])
            subject_rows.append({"split": split_name, "subject": int(subject), **m})
        tmp = df_sub[["subject", "segment_id", "class"]].copy().reset_index(drop=True)
        tmp["true_y"] = y_true
        tmp["pred_y"] = y_pred
        tmp["score_prob_nopain"] = base_scores
        tmp["score_final"] = final_scores
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
    plot_confusion(cm, ["NoPain", "Pain"], "Stage 1 Validation Confusion", out_dir / "confusion_validation.png")
    plot_per_subject(per_subject[per_subject["split"] == "validation"], "Stage 1 Validation Per-Subject Macro-F1", out_dir / "per_subject_validation.png")
    plot_calibration(val_pred["true_y"].to_numpy(), np.clip(val_pred["score_prob_nopain"].to_numpy(), 1e-6, 1 - 1e-6), "Stage 1 Calibration", out_dir / "calibration_validation.png")

    curve_df = maybe_xgb_curve(train_df, val_df, selected, spec, scaler_name=args.scaler)
    if not curve_df.empty:
        curve_df.to_csv(out_dir / "xgb_curve.csv", index=False)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(curve_df["iter"], curve_df["train_logloss"], label="train")
        ax.plot(curve_df["iter"], curve_df["val_logloss"], label="validation")
        ax.set_title("Stage 1 XGB Training Curve")
        ax.set_xlabel("round")
        ax.set_ylabel("logloss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "xgb_curve.png", dpi=140)
        plt.close(fig)

    report = [
        "# Stage 1 Run",
        f"- feature parquet: `{feature_fp}`",
        f"- normalization: `{args.norm}`",
        f"- feature set: `{args.feature_set}` ({len(selected)} features)",
        f"- model: `{spec.name}` {spec.params}",
        f"- scaler: `{args.scaler}`",
        f"- calibration: `{args.calibration}`",
        f"- anchor mode: `{args.anchor_mode}`",
        f"- anchor lambda: `{args.anchor_lambda}`",
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
    p.add_argument("--norm", choices=["subject_z", "subject_robust"], default="subject_robust")
    p.add_argument("--feature-set", choices=["bvp_only", "eda_only", "bvp_eda_core", "bvp_eda_resp_small", "eda_bvp_fwd8", "all"], default="bvp_eda_core")
    p.add_argument("--model", choices=["xgb", "rf", "logreg", "svm_linear", "svm_rbf"], default="xgb")
    p.add_argument("--scaler", choices=["std", "robust"], default="std")
    p.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--anchor-mode", choices=["none", "center", "z"], default="center")
    p.add_argument("--anchor-lambda", type=float, default=0.5)
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
    p.add_argument("--output-dir", default="results/final/stage1")
    args = p.parse_args()
    if args.model == "xgb" and not HAS_XGB:
        raise SystemExit("xgboost is not installed")
    main(args)
