from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.final_pipeline import (
    class_codes_3,
    exact_count_decode,
    metrics_multiclass,
    plot_confusion,
    plot_per_subject,
)


ROOT = Path(__file__).resolve().parents[1]


def load_run_combined_module():
    path = ROOT / "scripts" / "run_combined.py"
    spec = importlib.util.spec_from_file_location("run_combined_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RC = load_run_combined_module()


def default_feature_parquet(resample_tag: str) -> Path:
    return Path(f"results/tables/all_features_merged_{resample_tag}.parquet")


def align_frame_to_ref(ref_df: pd.DataFrame, src_df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    merged = ref_df[["segment_id", "subject", "class"]].merge(
        src_df[["segment_id"] + feat_cols], on="segment_id", how="left"
    )
    if merged[feat_cols].isna().any().any():
        raise RuntimeError("failed to align stage2 feature rows by segment_id")
    return merged


def predicted_anchor_map(df_sub: pd.DataFrame, score_nopain: np.ndarray) -> dict[int, list[str]]:
    out = {}
    for subject in sorted(df_sub["subject"].unique()):
        sub = df_sub[df_sub["subject"] == subject].reset_index(drop=True)
        p = score_nopain[df_sub["subject"] == subject]
        order = np.argsort(-p)[:12]
        out[int(subject)] = sub.iloc[order]["segment_id"].tolist()
    return out


def nopain_distance_confidence(df_ref: pd.DataFrame, df_stage2_full: pd.DataFrame, feat_cols: list[str], stage1_nopain_prob: np.ndarray, mode: str) -> np.ndarray:
    aligned = align_frame_to_ref(df_ref, df_stage2_full, feat_cols)
    anchor_map = predicted_anchor_map(df_ref, stage1_nopain_prob)
    conf = np.zeros(len(df_ref), dtype=np.float32)

    for subject in sorted(df_ref["subject"].unique()):
        mask = (aligned["subject"] == subject).to_numpy()
        sub = aligned.loc[mask].reset_index(drop=True)
        base = sub["segment_id"].isin(anchor_map[int(subject)])
        X = sub[feat_cols].to_numpy(dtype=np.float32)
        mu = sub.loc[base, feat_cols].mean(axis=0).to_numpy(dtype=np.float32)
        if mode == "center":
            dist = np.sqrt(np.mean((X - mu) ** 2, axis=1))
        elif mode == "z":
            sd = sub.loc[base, feat_cols].std(axis=0, ddof=0).to_numpy(dtype=np.float32)
            sd = np.where(sd > 1e-6, sd, 1.0)
            dist = np.mean(np.abs((X - mu) / sd), axis=1)
        else:
            raise ValueError(mode)
        z = (dist - np.median(dist)) / (np.std(dist) + 1e-6)
        c = 1.0 / (1.0 + np.exp(-z))
        conf[mask] = np.clip(c.astype(np.float32), 1e-6, 1.0 - 1e-6)
    return conf


def decode_subjectwise(df_sub: pd.DataFrame, stage1_probs: np.ndarray, stage2_probs: np.ndarray, variant: str, w0: float, w1: float, w2: float, conf: np.ndarray | None = None, alpha: float = 0.5) -> np.ndarray:
    pred = np.zeros(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        s1 = stage1_probs[mask]
        s2 = stage2_probs[mask]
        p_pain = np.clip(s1[:, 1], 1e-6, 1.0 - 1e-6)
        p_arm = np.clip(s2[:, 0], 1e-6, 1.0 - 1e-6)

        if variant == "baseline":
            log_scores = np.column_stack([
                w0 * np.log1p(-p_pain),
                w1 * np.log(p_pain) + w2 * np.log(p_arm),
                w1 * np.log(p_pain) + w2 * np.log1p(-p_arm),
            ])
        elif variant.startswith("gate_"):
            assert conf is not None
            c = conf[mask]
            gain = 0.5 + c
            log_scores = np.column_stack([
                w0 * np.log1p(-p_pain),
                w1 * np.log(p_pain) + (w2 * gain) * np.log(p_arm),
                w1 * np.log(p_pain) + (w2 * gain) * np.log1p(-p_arm),
            ])
        elif variant.startswith("painaux_"):
            assert conf is not None
            c = conf[mask]
            log_scores = np.column_stack([
                w0 * np.log1p(-p_pain) + alpha * np.log1p(-c),
                w1 * np.log(p_pain) + w2 * np.log(p_arm) + alpha * np.log(c),
                w1 * np.log(p_pain) + w2 * np.log1p(-p_arm) + alpha * np.log(c),
            ])
        else:
            raise ValueError(variant)

        log_scores = np.nan_to_num(log_scores, nan=-1e9, neginf=-1e9, posinf=1e9)
        pred[mask] = exact_count_decode(log_scores, [12, 12, 12])
    return pred


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_fp = args.feature_parquet or default_feature_parquet(args.resample_tag)
    print(f"[load] feature parquet: {feature_fp}")
    df_all, feat_cols = RC.load_clean_features(feature_fp)
    norm_map = RC.build_norm_map(df_all, feat_cols)

    s1_norm_df = norm_map[args.stage1_norm]
    train_stage1_for_fs = s1_norm_df[s1_norm_df["split"] == "train"].reset_index(drop=True)
    s1_features = RC.stage1_feature_sets(train_stage1_for_fs, feat_cols)[args.stage1_feature_set]
    s1_spec = RC.build_stage1_spec(args)
    print(f"[fit] stage1: {args.stage1_norm} | {args.stage1_feature_set} | {s1_spec.name}")
    train_stage1, val_stage1, s1_train_probs, s1_val_probs = RC.fit_stage1(
        s1_norm_df,
        s1_features,
        s1_spec,
        scaler_name=args.stage1_scaler,
        calibration=args.stage1_calibration,
        anchor_mode=args.stage1_anchor_mode,
        anchor_lambda=args.stage1_anchor_lambda,
        df_global=df_all,
    )

    s2_norm_df = norm_map[args.stage2_norm]
    train_stage2_for_fs = s2_norm_df[s2_norm_df["split"] == "train"].reset_index(drop=True)
    s2_features = RC.stage2_feature_sets(train_stage2_for_fs, feat_cols)[args.stage2_feature_set]
    s2_spec = RC.build_stage2_spec(args)
    print(f"[fit] stage2: {args.stage2_norm} | {args.stage2_feature_set} | {s2_spec.name}")
    train_stage2, val_stage2, s2_train_probs, s2_val_probs = RC.fit_stage2(
        s2_norm_df,
        df_all,
        s2_features,
        s2_spec,
        scaler_name=args.stage2_scaler,
        calibration=args.stage2_calibration,
        anchor_mode=args.stage2_anchor_mode,
        train_stage1_scores=s1_train_probs[:, 0],
        val_stage1_scores=s1_val_probs[:, 0],
    )

    s2_train_probs = RC.align_probs_to_ref(train_stage1, train_stage2, s2_train_probs)
    s2_val_probs = RC.align_probs_to_ref(val_stage1, val_stage2, s2_val_probs)

    conf_train_center = nopain_distance_confidence(train_stage1, train_stage2, s2_features, s1_train_probs[:, 0], mode="center")
    conf_val_center = nopain_distance_confidence(val_stage1, val_stage2, s2_features, s1_val_probs[:, 0], mode="center")
    conf_train_z = nopain_distance_confidence(train_stage1, train_stage2, s2_features, s1_train_probs[:, 0], mode="z")
    conf_val_z = nopain_distance_confidence(val_stage1, val_stage2, s2_features, s1_val_probs[:, 0], mode="z")

    variants = [
        ("baseline", None, None),
        ("gate_center", conf_train_center, conf_val_center),
        ("gate_z", conf_train_z, conf_val_z),
        ("painaux_center", conf_train_center, conf_val_center),
        ("painaux_z", conf_train_z, conf_val_z),
    ]

    summary_rows = []
    per_subject_rows = []
    pred_rows = []

    for variant, conf_train, conf_val in tqdm(variants, desc="nopain-stage2 variants"):
        for split_name, df_sub, s1_probs, s2_probs, conf in (
            ("train_loso", train_stage1, s1_train_probs, s2_train_probs, conf_train),
            ("validation", val_stage1, s1_val_probs, s2_val_probs, conf_val),
        ):
            y_true = class_codes_3(df_sub["class"])
            y_pred = decode_subjectwise(df_sub, s1_probs, s2_probs, variant=variant, w0=args.w0, w1=args.w1, w2=args.w2, conf=conf, alpha=args.alpha)
            met = metrics_multiclass(y_true, y_pred)
            summary_rows.append({"variant": variant, "split": split_name, **met})
            for subject in sorted(df_sub["subject"].unique()):
                mask = (df_sub["subject"] == subject).to_numpy()
                m = metrics_multiclass(y_true[mask], y_pred[mask])
                per_subject_rows.append({"variant": variant, "split": split_name, "subject": int(subject), **m})
            tmp = df_sub[["subject", "segment_id", "class"]].copy().reset_index(drop=True)
            tmp["variant"] = variant
            tmp["split"] = split_name
            tmp["true_y"] = y_true
            tmp["pred_y"] = y_pred
            tmp["stage1_nopain_prob"] = s1_probs[:, 0]
            tmp["stage2_arm_prob"] = s2_probs[:, 0]
            if conf is not None:
                tmp["nopain_conf"] = conf
            pred_rows.append(tmp)

    summary = pd.DataFrame(summary_rows).sort_values(["split", "macro_f1"], ascending=[True, False]).reset_index(drop=True)
    per_subject = pd.DataFrame(per_subject_rows)
    preds = pd.concat(pred_rows, ignore_index=True)

    summary.to_csv(out_dir / "summary.csv", index=False)
    per_subject.to_csv(out_dir / "per_subject.csv", index=False)
    preds.to_parquet(out_dir / "predictions.parquet", index=False)

    best_val_variant = summary[summary["split"] == "validation"].iloc[0]["variant"]
    val_best = preds[(preds["split"] == "validation") & (preds["variant"] == best_val_variant)].copy()
    cm = pd.crosstab(val_best["true_y"], val_best["pred_y"]).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
    plot_confusion(cm, ["NoPain", "Arm", "Hand"], f"Validation Confusion: {best_val_variant}", out_dir / "best_confusion.png")
    plot_per_subject(
        per_subject[(per_subject["split"] == "validation") & (per_subject["variant"] == best_val_variant)],
        f"Validation Per-Subject Macro-F1: {best_val_variant}",
        out_dir / "best_per_subject.png",
    )

    report = [
        "# NoPain-Informed Stage-2 Calibration Check",
        f"- feature parquet: `{feature_fp}`",
        f"- baseline stage1: `{args.stage1_norm} | {args.stage1_feature_set} | {args.stage1_model}`",
        f"- baseline stage2: `{args.stage2_norm} | {args.stage2_feature_set} | {args.stage2_model}`",
        f"- decoder weights: w0={args.w0}, w1={args.w1}, w2={args.w2}`",
        f"- painaux alpha: `{args.alpha}`",
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

    p.add_argument("--stage1-norm", choices=["subject_z", "subject_robust"], default="subject_robust")
    p.add_argument("--stage1-feature-set", choices=["bvp_only", "eda_only", "bvp_eda_core", "bvp_eda_resp_small", "all", "all_raw"], default="bvp_eda_core")
    p.add_argument("--stage1-model", choices=["xgb", "rf", "logreg", "svm_linear", "svm_rbf"], default="xgb")
    p.add_argument("--stage1-scaler", choices=["std", "robust"], default="std")
    p.add_argument("--stage1-calibration", choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--stage1-anchor-mode", choices=["none", "center", "z"], default="center")
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
    p.add_argument("--stage2-feature-set", choices=["resp_all", "resp_top20", "resp_bvp5"], default="resp_all")
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

    p.add_argument("--w0", type=float, default=1.0)
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--output-dir", default="results/final/nopain_stage2_check")
    args = p.parse_args()
    main(args)
