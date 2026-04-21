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
    class_codes_3,
    decode_joint_weighted,
    decode_split_topk,
    exact12_binary_predictions,
    fit_binary_calibrator,
    fit_binary_proba,
    load_clean_features,
    metrics_multiclass,
    plot_calibration,
    plot_confusion,
    plot_per_subject,
    stage1_anchor_scores,
    stage1_feature_sets,
    stage2_feature_sets,
)


def default_feature_parquet(resample_tag: str) -> Path:
    return Path(f"results/tables/all_features_merged_{resample_tag}.parquet")


def build_stage1_spec(args: argparse.Namespace) -> ModelSpec:
    if args.stage1_model == "xgb":
        return ModelSpec("xgb", {
            "n_estimators": args.stage1_xgb_n_estimators,
            "max_depth": args.stage1_xgb_max_depth,
            "learning_rate": args.stage1_xgb_learning_rate,
            "subsample": args.stage1_xgb_subsample,
            "colsample_bytree": args.stage1_xgb_colsample_bytree,
        })
    if args.stage1_model == "rf":
        return ModelSpec("rf", {"n_estimators": args.stage1_rf_n_estimators, "max_depth": args.stage1_rf_max_depth})
    if args.stage1_model == "logreg":
        return ModelSpec("logreg", {"C": args.stage1_logreg_c})
    if args.stage1_model == "svm_linear":
        return ModelSpec("svm_linear", {"C": args.stage1_svm_c})
    if args.stage1_model == "svm_rbf":
        return ModelSpec("svm_rbf", {"C": args.stage1_svm_c, "gamma": args.stage1_svm_gamma})
    raise ValueError(args.stage1_model)


def build_stage2_spec(args: argparse.Namespace) -> ModelSpec:
    if args.stage2_model == "xgb":
        return ModelSpec("xgb", {
            "n_estimators": args.stage2_xgb_n_estimators,
            "max_depth": args.stage2_xgb_max_depth,
            "learning_rate": args.stage2_xgb_learning_rate,
            "subsample": args.stage2_xgb_subsample,
            "colsample_bytree": args.stage2_xgb_colsample_bytree,
        })
    if args.stage2_model == "rf":
        return ModelSpec("rf", {"n_estimators": args.stage2_rf_n_estimators, "max_depth": args.stage2_rf_max_depth})
    if args.stage2_model == "logreg":
        return ModelSpec("logreg", {"C": args.stage2_logreg_c})
    if args.stage2_model == "svm_linear":
        return ModelSpec("svm_linear", {"C": args.stage2_svm_c})
    if args.stage2_model == "svm_rbf":
        return ModelSpec("svm_rbf", {"C": args.stage2_svm_c, "gamma": args.stage2_svm_gamma})
    raise ValueError(args.stage2_model)


def fit_stage1(df_norm: pd.DataFrame, feat_cols: list[str], spec: ModelSpec, scaler_name: str, calibration: str, anchor_mode: str, anchor_lambda: float, df_global: pd.DataFrame):
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    train_probs_raw = np.zeros((len(train), 2), dtype=np.float32)
    for subject in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subject].reset_index(drop=True)
        te = train[train["subject"] == subject].reset_index(drop=True)
        probs = fit_binary_proba(tr, te, feat_cols, (tr["class"] != "NoPain").astype(int).to_numpy(), scaler_name=scaler_name, spec=spec)
        train_probs_raw[train["subject"] == subject] = probs
    val_probs_raw = fit_binary_proba(train, val, feat_cols, (train["class"] != "NoPain").astype(int).to_numpy(), scaler_name=scaler_name, spec=spec)

    cal = fit_binary_calibrator(calibration, train_probs_raw[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    train_nop = cal(train_probs_raw[:, 0])
    val_nop = cal(val_probs_raw[:, 0])
    if anchor_mode != "none":
        global_train = df_global[df_global["split"] == "train"].reset_index(drop=True)
        global_val = df_global[df_global["split"] == "validation"].reset_index(drop=True)
        train_final = stage1_anchor_scores(train, global_train, feat_cols, train_nop, mode=anchor_mode, lam=anchor_lambda)
        val_final = stage1_anchor_scores(val, global_val, feat_cols, val_nop, mode=anchor_mode, lam=anchor_lambda)
        train_final = 1.0 / (1.0 + np.exp(-train_final))
        val_final = 1.0 / (1.0 + np.exp(-val_final))
    else:
        train_final = train_nop
        val_final = val_nop
    train_probs = np.column_stack([np.clip(train_final, 1e-6, 1 - 1e-6), 1 - np.clip(train_final, 1e-6, 1 - 1e-6)]).astype(np.float32)
    val_probs = np.column_stack([np.clip(val_final, 1e-6, 1 - 1e-6), 1 - np.clip(val_final, 1e-6, 1 - 1e-6)]).astype(np.float32)
    return train, val, train_probs, val_probs


def predicted_anchor_map(df_sub: pd.DataFrame, score_nopain: np.ndarray) -> dict[int, list[str]]:
    out = {}
    for subject in sorted(df_sub["subject"].unique()):
        sub = df_sub[df_sub["subject"] == subject].reset_index(drop=True)
        p = score_nopain[df_sub["subject"] == subject]
        order = np.argsort(-p)[:12]
        out[int(subject)] = sub.iloc[order]["segment_id"].tolist()
    return out


def adapt_with_anchor(df_norm: pd.DataFrame, df_global: pd.DataFrame, feat_cols: list[str], anchor_map: dict[int, list[str]], mode: str) -> pd.DataFrame:
    out = df_norm.copy()
    for subject in sorted(out["subject"].unique()):
        rows = out["subject"] == subject
        gsub = df_global[df_global["subject"] == subject].reset_index(drop=True)
        base = gsub["segment_id"].isin(anchor_map.get(int(subject), []))
        if int(base.sum()) == 0:
            continue
        mu = gsub.loc[base, feat_cols].mean(axis=0)
        if mode == "center":
            out.loc[rows, feat_cols] = (out.loc[rows, feat_cols] - mu).astype(np.float32)
        elif mode == "z":
            sd = gsub.loc[base, feat_cols].std(axis=0, ddof=0)
            sd = sd.where(sd > 0, 1.0)
            out.loc[rows, feat_cols] = ((out.loc[rows, feat_cols] - mu) / sd).astype(np.float32)
        else:
            raise ValueError(mode)
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out


def align_probs_to_ref(ref_df: pd.DataFrame, src_df: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    prob_df = pd.DataFrame({
        "segment_id": src_df["segment_id"].to_numpy(),
        "p0": probs[:, 0],
        "p1": probs[:, 1],
    })
    merged = ref_df[["segment_id"]].merge(prob_df, on="segment_id", how="left")
    if merged[["p0", "p1"]].isna().any().any():
        raise RuntimeError("failed to align stage probabilities by segment_id")
    return merged[["p0", "p1"]].to_numpy(dtype=np.float32)


def fit_stage2(df_norm: pd.DataFrame, df_global: pd.DataFrame, feat_cols: list[str], spec: ModelSpec, scaler_name: str, calibration: str, anchor_mode: str, train_stage1_scores: np.ndarray, val_stage1_scores: np.ndarray):
    train_full = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val_full = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    if anchor_mode != "none":
        global_train = df_global[df_global["split"] == "train"].reset_index(drop=True)
        global_val = df_global[df_global["split"] == "validation"].reset_index(drop=True)
        train_full = adapt_with_anchor(train_full, global_train, feat_cols, predicted_anchor_map(train_full, train_stage1_scores), mode=anchor_mode)
        val_full = adapt_with_anchor(val_full, global_val, feat_cols, predicted_anchor_map(val_full, val_stage1_scores), mode=anchor_mode)

    train = train_full[train_full["class"].isin(ARM_HAND)].reset_index(drop=True)
    val = val_full[val_full["class"].isin(ARM_HAND)].reset_index(drop=True)
    train_probs_raw = np.zeros((len(train_full), 2), dtype=np.float32)
    for subject in sorted(train["subject"].unique()):
        tr = train[train["subject"] != subject].reset_index(drop=True)
        te_full = train_full[train_full["subject"] == subject].reset_index(drop=True)
        probs = fit_binary_proba(tr, te_full, feat_cols, armhand_binary(tr["class"]), scaler_name=scaler_name, spec=spec)
        train_probs_raw[train_full["subject"] == subject] = probs

    val_probs_raw = fit_binary_proba(train, val_full, feat_cols, armhand_binary(train["class"]), scaler_name=scaler_name, spec=spec)
    cal = fit_binary_calibrator(calibration, train_probs_raw[(train_full["class"] != "NoPain").to_numpy(), 0], (train_full[train_full["class"] != "NoPain"]["class"] == "PainArm").astype(int).to_numpy())
    train_arm = cal(train_probs_raw[:, 0])
    val_arm = cal(val_probs_raw[:, 0])
    train_probs = np.column_stack([np.clip(train_arm, 1e-6, 1 - 1e-6), 1 - np.clip(train_arm, 1e-6, 1 - 1e-6)]).astype(np.float32)
    val_probs = np.column_stack([np.clip(val_arm, 1e-6, 1 - 1e-6), 1 - np.clip(val_arm, 1e-6, 1 - 1e-6)]).astype(np.float32)
    return train_full, val_full, train_probs, val_probs


def decode_subjectwise(df_sub: pd.DataFrame, stage1_probs: np.ndarray, stage2_probs: np.ndarray, decoder: str, w0: float, w1: float, w2: float) -> np.ndarray:
    y_pred = np.zeros(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        if decoder == "joint_weighted":
            y_pred[mask] = decode_joint_weighted(stage1_probs[mask], stage2_probs[mask], w0=w0, w1=w1, w2=w2)
        else:
            y_pred[mask] = decode_split_topk(stage1_probs[mask], stage2_probs[mask])
    return y_pred


def main(args: argparse.Namespace) -> None:
    feature_fp = args.feature_parquet or default_feature_parquet(args.resample_tag)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features(feature_fp)
    norm_map = build_norm_map(df_all, feat_cols)

    stage1_norm_df = norm_map[args.stage1_norm]
    train_stage1_for_fs = stage1_norm_df[stage1_norm_df["split"] == "train"].reset_index(drop=True)
    s1_features = stage1_feature_sets(train_stage1_for_fs, feat_cols)[args.stage1_feature_set]
    s1_spec = build_stage1_spec(args)
    train_stage1, val_stage1, s1_train_probs, s1_val_probs = fit_stage1(stage1_norm_df, s1_features, s1_spec, scaler_name=args.stage1_scaler, calibration=args.stage1_calibration, anchor_mode=args.stage1_anchor_mode, anchor_lambda=args.stage1_anchor_lambda, df_global=df_all)

    stage2_norm_df = norm_map[args.stage2_norm]
    train_stage2_for_fs = stage2_norm_df[stage2_norm_df["split"] == "train"].reset_index(drop=True)
    s2_features = stage2_feature_sets(train_stage2_for_fs, feat_cols)[args.stage2_feature_set]
    s2_spec = build_stage2_spec(args)
    train_stage2, val_stage2, s2_train_probs, s2_val_probs = fit_stage2(stage2_norm_df, df_all, s2_features, s2_spec, scaler_name=args.stage2_scaler, calibration=args.stage2_calibration, anchor_mode=args.stage2_anchor_mode, train_stage1_scores=s1_train_probs[:, 0], val_stage1_scores=s1_val_probs[:, 0])

    s2_train_probs = align_probs_to_ref(train_stage1, train_stage2, s2_train_probs)
    s2_val_probs = align_probs_to_ref(val_stage1, val_stage2, s2_val_probs)

    rows = []
    subject_rows = []
    pred_rows = []
    for split_name, df_sub, stage1_probs, stage2_probs in (("train_loso", train_stage1, s1_train_probs, s2_train_probs), ("validation", val_stage1, s1_val_probs, s2_val_probs)):
        y_pred = decode_subjectwise(df_sub, stage1_probs, stage2_probs, decoder=args.decoder, w0=args.w0, w1=args.w1, w2=args.w2)
        y_true = class_codes_3(df_sub["class"])
        met = metrics_multiclass(y_true, y_pred)
        rows.append({"split": split_name, **met})
        for subject in sorted(df_sub["subject"].unique()):
            mask = (df_sub["subject"] == subject).to_numpy()
            m = metrics_multiclass(y_true[mask], y_pred[mask])
            subject_rows.append({"split": split_name, "subject": int(subject), **m})
        tmp = df_sub[["subject", "segment_id", "class"]].copy().reset_index(drop=True)
        tmp["pred_class"] = ["NoPain" if i == 0 else ("PainArm" if i == 1 else "PainHand") for i in y_pred]
        tmp["true_y"] = y_true
        tmp["pred_y"] = y_pred
        tmp["stage1_nopain_prob"] = stage1_probs[:, 0]
        tmp["stage2_arm_prob"] = stage2_probs[:, 0]
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
    plot_confusion(cm, ["NoPain", "Arm", "Hand"], "Combined Validation Confusion", out_dir / "confusion_validation.png")
    plot_per_subject(per_subject[per_subject["split"] == "validation"], "Combined Validation Per-Subject Macro-F1", out_dir / "per_subject_validation.png")
    plot_calibration((val_pred["class"] == "NoPain").astype(int).to_numpy(), np.clip(val_pred["stage1_nopain_prob"].to_numpy(), 1e-6, 1 - 1e-6), "Stage 1 Calibration (Combined)", out_dir / "calibration_stage1.png")
    plot_calibration((val_pred[val_pred["class"] != "NoPain"]["class"] == "PainArm").astype(int).to_numpy(), np.clip(val_pred[val_pred["class"] != "NoPain"]["stage2_arm_prob"].to_numpy(), 1e-6, 1 - 1e-6), "Stage 2 Calibration (Combined)", out_dir / "calibration_stage2.png")

    report = [
        "# Combined Run",
        f"- feature parquet: `{feature_fp}`",
        f"- stage1 norm: `{args.stage1_norm}`",
        f"- stage1 feature set: `{args.stage1_feature_set}` ({len(s1_features)} features)",
        f"- stage1 model: `{s1_spec.name}` {s1_spec.params}",
        f"- stage1 calibration: `{args.stage1_calibration}`",
        f"- stage1 anchor: `{args.stage1_anchor_mode}` lambda={args.stage1_anchor_lambda}",
        f"- stage2 norm: `{args.stage2_norm}`",
        f"- stage2 feature set: `{args.stage2_feature_set}` ({len(s2_features)} features)",
        f"- stage2 model: `{s2_spec.name}` {s2_spec.params}",
        f"- stage2 calibration: `{args.stage2_calibration}`",
        f"- stage2 anchor: `{args.stage2_anchor_mode}`",
        f"- decoder: `{args.decoder}`",
        f"- weights: w0={args.w0}, w1={args.w1}, w2={args.w2}",
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

    p.add_argument("--stage1-norm", choices=["subject_z", "subject_robust"], default="subject_z")
    p.add_argument("--stage1-feature-set", choices=["bvp_only", "eda_only", "bvp_eda_core", "bvp_eda_resp_small", "eda_bvp_fwd8", "all", "all_raw"], default="bvp_eda_resp_small")
    p.add_argument("--stage1-model", choices=["xgb", "rf", "logreg", "svm_linear", "svm_rbf"], default="logreg")
    p.add_argument("--stage1-scaler", choices=["std", "robust"], default="std")
    p.add_argument("--stage1-calibration", choices=["none", "sigmoid", "isotonic"], default="isotonic")
    p.add_argument("--stage1-anchor-mode", choices=["none", "center", "z"], default="none")
    p.add_argument("--stage1-anchor-lambda", type=float, default=0.5)
    p.add_argument("--stage1-xgb-n-estimators", type=int, default=200)
    p.add_argument("--stage1-xgb-max-depth", type=int, default=4)
    p.add_argument("--stage1-xgb-learning-rate", type=float, default=0.08)
    p.add_argument("--stage1-xgb-subsample", type=float, default=1.0)
    p.add_argument("--stage1-xgb-colsample-bytree", type=float, default=1.0)
    p.add_argument("--stage1-rf-n-estimators", type=int, default=400)
    p.add_argument("--stage1-rf-max-depth", default=None)
    p.add_argument("--stage1-logreg-c", type=float, default=1.0)
    p.add_argument("--stage1-svm-c", type=float, default=3.0)
    p.add_argument("--stage1-svm-gamma", default="scale")

    p.add_argument("--stage2-norm", choices=["global", "subject_z", "subject_robust"], default="subject_z")
    p.add_argument("--stage2-feature-set", choices=["resp_all", "resp_top20", "resp_bvp5", "eda_resp_top30", "bvp_resp_top30", "all_top40"], default="bvp_resp_top30")
    p.add_argument("--stage2-model", choices=["xgb", "rf", "logreg", "svm_linear", "svm_rbf"], default="logreg")
    p.add_argument("--stage2-scaler", choices=["std", "robust"], default="robust")
    p.add_argument("--stage2-calibration", choices=["none", "sigmoid", "isotonic"], default="isotonic")
    p.add_argument("--stage2-anchor-mode", choices=["none", "center", "z"], default="none")
    p.add_argument("--stage2-xgb-n-estimators", type=int, default=200)
    p.add_argument("--stage2-xgb-max-depth", type=int, default=4)
    p.add_argument("--stage2-xgb-learning-rate", type=float, default=0.08)
    p.add_argument("--stage2-xgb-subsample", type=float, default=1.0)
    p.add_argument("--stage2-xgb-colsample-bytree", type=float, default=1.0)
    p.add_argument("--stage2-rf-n-estimators", type=int, default=400)
    p.add_argument("--stage2-rf-max-depth", default=None)
    p.add_argument("--stage2-logreg-c", type=float, default=1.0)
    p.add_argument("--stage2-svm-c", type=float, default=3.0)
    p.add_argument("--stage2-svm-gamma", default="scale")

    p.add_argument("--decoder", choices=["joint_weighted", "split_topk"], default="joint_weighted")
    p.add_argument("--w0", type=float, default=1.0)
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=1.0)
    p.add_argument("--output-dir", default="results/final/combined")
    args = p.parse_args()
    if (args.stage1_model == "xgb" or args.stage2_model == "xgb") and not HAS_XGB:
        raise SystemExit("xgboost is not installed")
    main(args)
