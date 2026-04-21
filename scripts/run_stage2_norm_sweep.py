"""
Stage 2 per-subject normalization sweep.

Tests whether subject amplitude variability is hurting LOSO by comparing:
  - global                : no per-subject norm
  - subject_z             : per-subject mean/std over ALL rows (existing)
  - subject_robust        : per-subject median/IQR over ALL rows (existing)
  - nopain_center         : subtract per-subject NoPain mean only
  - nopain_z              : subtract NoPain mean, divide by NoPain std
  - nopain_robust         : subtract NoPain median, divide by NoPain IQR
  - nopain_z_responder    : nopain_z + per-subject responder-strength scalar covariate

Responder strength = ||mean(Pain_rows) - mean(NoPain_rows)||_2 computed on
globally z-scored features, one scalar per subject, broadcast as a feature
column. Captures whether a subject is a strong/weak responder.

Runs LOSO arm-vs-hand binary on pain-only rows using best-model defaults
(logreg + resp_top20 + robust scaler + isotonic cal). Also computes
validation macro-F1 (train on all train, predict val).
"""
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
    fit_binary_calibrator,
    fit_binary_proba,
    load_clean_features,
    metrics_binary,
    stage2_feature_sets,
)

RESPONDER_COL = "_responder_strength"

NORM_CHOICES = [
    "global",
    "subject_z",
    "subject_robust",
    "nopain_center",
    "nopain_z",
    "nopain_robust",
    "nopain_z_responder",
]


def _global_z(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    tr_mask = df["split"] == "train"
    mu = df.loc[tr_mask, feat_cols].mean(axis=0)
    sd = df.loc[tr_mask, feat_cols].std(axis=0, ddof=0).replace(0.0, 1.0)
    out = df.copy()
    out[feat_cols] = ((out[feat_cols] - mu) / sd).fillna(0.0).astype(np.float32)
    return out


def _responder_strength(df_all: pd.DataFrame, feat_cols: list[str]) -> pd.Series:
    z = _global_z(df_all, feat_cols)
    rows = []
    for subj, sub in z.groupby("subject"):
        pain = sub[sub["class"].isin(ARM_HAND)][feat_cols].mean(axis=0)
        nopain = sub[sub["class"] == "NoPain"][feat_cols].mean(axis=0)
        if pain.isna().any() or nopain.isna().any():
            rows.append((subj, 0.0))
            continue
        strength = float(np.linalg.norm((pain - nopain).to_numpy()))
        rows.append((subj, strength))
    return pd.Series(dict(rows), name=RESPONDER_COL)


def norm_subject_mixed(df: pd.DataFrame, feat_cols: list[str], how: str) -> pd.DataFrame:
    out = df.copy()
    if how == "z":
        mu = out.groupby("subject")[feat_cols].transform("mean")
        sd = out.groupby("subject")[feat_cols].transform("std", ddof=0)
        sd = sd.where(sd > 0, 1.0)
        out[feat_cols] = ((out[feat_cols] - mu) / sd).fillna(0.0).astype(np.float32)
    elif how == "robust":
        med = out.groupby("subject")[feat_cols].transform("median")
        q75 = out.groupby("subject")[feat_cols].transform(lambda s: s.quantile(0.75))
        q25 = out.groupby("subject")[feat_cols].transform(lambda s: s.quantile(0.25))
        iqr = q75 - q25
        iqr = iqr.where(iqr > 0, 1.0)
        out[feat_cols] = ((out[feat_cols] - med) / iqr).fillna(0.0).astype(np.float32)
    else:
        raise ValueError(how)
    return out


def norm_nopain_anchor(df: pd.DataFrame, feat_cols: list[str], how: str) -> pd.DataFrame:
    """Per-subject norm using only NoPain rows as baseline stats."""
    out = df.copy()
    for subj, sub in df.groupby("subject"):
        base = sub[sub["class"] == "NoPain"][feat_cols]
        if base.empty:
            continue
        idx = sub.index
        if how == "center":
            mu = base.mean(axis=0)
            out.loc[idx, feat_cols] = (df.loc[idx, feat_cols] - mu).astype(np.float32)
        elif how == "z":
            mu = base.mean(axis=0)
            sd = base.std(axis=0, ddof=0)
            sd = sd.where(sd > 0, 1.0)
            out.loc[idx, feat_cols] = ((df.loc[idx, feat_cols] - mu) / sd).astype(np.float32)
        elif how == "robust":
            med = base.median(axis=0)
            q75 = base.quantile(0.75)
            q25 = base.quantile(0.25)
            iqr = q75 - q25
            iqr = iqr.where(iqr > 0, 1.0)
            out.loc[idx, feat_cols] = ((df.loc[idx, feat_cols] - med) / iqr).astype(np.float32)
        else:
            raise ValueError(how)
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out


def apply_norm(df: pd.DataFrame, feat_cols: list[str], strategy: str) -> tuple[pd.DataFrame, list[str]]:
    """Return (normed_df, feat_cols_to_use). feat_cols may gain responder_strength."""
    if strategy == "global":
        return df.copy(), feat_cols
    if strategy == "subject_z":
        return norm_subject_mixed(df, feat_cols, "z"), feat_cols
    if strategy == "subject_robust":
        return norm_subject_mixed(df, feat_cols, "robust"), feat_cols
    if strategy == "nopain_center":
        return norm_nopain_anchor(df, feat_cols, "center"), feat_cols
    if strategy == "nopain_z":
        return norm_nopain_anchor(df, feat_cols, "z"), feat_cols
    if strategy == "nopain_robust":
        return norm_nopain_anchor(df, feat_cols, "robust"), feat_cols
    if strategy == "nopain_z_responder":
        normed = norm_nopain_anchor(df, feat_cols, "z")
        strength = _responder_strength(df, feat_cols)
        normed[RESPONDER_COL] = normed["subject"].map(strength).astype(np.float32).fillna(0.0)
        return normed, feat_cols + [RESPONDER_COL]
    raise ValueError(strategy)


def exact12_arm_predictions(df_sub: pd.DataFrame, arm_scores: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-arm_scores[mask])]
        arm_idx = set(order[:12].tolist())
        pred[mask] = [0 if i in arm_idx else 1 for i in idx]
    return pred


def build_model_spec(model: str) -> ModelSpec:
    if model == "logreg":
        return ModelSpec("logreg", {"C": 1.0})
    if model == "xgb":
        return ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08,
                                  "subsample": 1.0, "colsample_bytree": 1.0})
    if model == "rf":
        return ModelSpec("rf", {"n_estimators": 400, "max_depth": None})
    raise ValueError(model)


def run_one(df_normed: pd.DataFrame, feat_cols: list[str], feature_set: str, model: str,
            scaler: str, calibration: str) -> dict:
    train_df = df_normed[df_normed["split"] == "train"].reset_index(drop=True)
    val_df = df_normed[df_normed["split"] == "validation"].reset_index(drop=True)

    base_feature_sets = stage2_feature_sets(train_df, [c for c in feat_cols if c != RESPONDER_COL])
    selected = list(base_feature_sets[feature_set])
    if RESPONDER_COL in feat_cols:
        selected = selected + [RESPONDER_COL]

    spec = build_model_spec(model)
    train_pain = train_df[train_df["class"].isin(ARM_HAND)].reset_index(drop=True)
    val_pain = val_df[val_df["class"].isin(ARM_HAND)].reset_index(drop=True)

    train_probs = np.zeros((len(train_pain), 2), dtype=np.float32)
    for subject in sorted(train_pain["subject"].unique()):
        tr = train_pain[train_pain["subject"] != subject].reset_index(drop=True)
        te = train_pain[train_pain["subject"] == subject].reset_index(drop=True)
        probs = fit_binary_proba(tr, te, selected, armhand_binary(tr["class"]), scaler_name=scaler, spec=spec)
        train_probs[train_pain["subject"] == subject] = probs
    val_probs = fit_binary_proba(train_pain, val_pain, selected, armhand_binary(train_pain["class"]),
                                  scaler_name=scaler, spec=spec)

    cal = fit_binary_calibrator(calibration, train_probs[:, 0],
                                 (train_pain["class"] == "PainArm").astype(int).to_numpy())
    train_arm = cal(train_probs[:, 0])
    val_arm = cal(val_probs[:, 0])

    results = {}
    per_subject = {}
    for name, sub, scores in (("loso", train_pain, train_arm), ("val", val_pain, val_arm)):
        y_true = (sub["class"] == "PainHand").astype(int).to_numpy()
        y_pred = exact12_arm_predictions(sub, scores)
        m = metrics_binary(y_true, y_pred)
        results[f"{name}_macro_f1"] = m["macro_f1"]
        results[f"{name}_acc"] = m["accuracy"]
        sub_f1s = []
        for subj in sorted(sub["subject"].unique()):
            mask = (sub["subject"] == subj).to_numpy()
            mm = metrics_binary(y_true[mask], y_pred[mask])
            sub_f1s.append(mm["macro_f1"])
            per_subject.setdefault(int(subj), {})[f"{name}_macro_f1"] = mm["macro_f1"]
        results[f"{name}_per_subject_std"] = float(np.std(sub_f1s))
        results[f"{name}_per_subject_min"] = float(np.min(sub_f1s))
    return {"summary": results, "per_subject": per_subject, "n_features": len(selected)}


def main(args: argparse.Namespace) -> None:
    feature_fp = args.feature_parquet or Path(f"results/tables/all_features_merged_{args.resample_tag}.parquet")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {feature_fp}")
    df_all, feat_cols = load_clean_features(feature_fp)
    print(f"[load] rows={len(df_all)} features={len(feat_cols)} subjects={df_all['subject'].nunique()}")

    strategies = args.strategies or NORM_CHOICES
    models = args.models

    summary_rows = []
    per_subj_rows = []
    for strat in strategies:
        for model in models:
            print(f"[run] strategy={strat} model={model}")
            normed, cols = apply_norm(df_all, feat_cols, strat)
            out = run_one(normed, cols, args.feature_set, model, args.scaler, args.calibration)
            row = {"strategy": strat, "model": model, "n_features": out["n_features"], **out["summary"]}
            summary_rows.append(row)
            print(f"    loso_macro_f1={row['loso_macro_f1']:.4f} "
                  f"val_macro_f1={row['val_macro_f1']:.4f} "
                  f"per_subj_std={row['loso_per_subject_std']:.3f} "
                  f"per_subj_min={row['loso_per_subject_min']:.3f}")
            for subj, d in out["per_subject"].items():
                per_subj_rows.append({"strategy": strat, "model": model, "subject": subj, **d})

    summary = pd.DataFrame(summary_rows).sort_values("loso_macro_f1", ascending=False)
    per_subj = pd.DataFrame(per_subj_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_subj.to_csv(out_dir / "per_subject.csv", index=False)

    report = ["# Stage 2 Normalization Sweep", "",
              f"- features: `{feature_fp}`",
              f"- feature set: `{args.feature_set}`",
              f"- models: {models}",
              f"- scaler: `{args.scaler}`  calibration: `{args.calibration}`",
              "", "## Results (sorted by LOSO macro-F1)", "",
              summary.to_markdown(index=False, floatfmt=".4f")]
    (out_dir / "report.md").write_text("\n".join(report))
    print(f"[done] wrote {out_dir}/summary.csv, per_subject.csv, report.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature-parquet", type=Path, default=None)
    p.add_argument("--resample-tag", default="1022")
    p.add_argument("--strategies", nargs="*", choices=NORM_CHOICES, default=None,
                    help="subset of normalization strategies to run (default: all)")
    p.add_argument("--models", nargs="+", choices=["logreg", "xgb", "rf"], default=["logreg"])
    p.add_argument("--feature-set", choices=["resp_all", "resp_top20", "resp_bvp5",
                                               "eda_resp_top30", "bvp_resp_top30", "all_top40"],
                    default="resp_top20")
    p.add_argument("--scaler", choices=["std", "robust"], default="robust")
    p.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], default="isotonic")
    p.add_argument("--output-dir", default="results/final/stage2_norm_sweep")
    args = p.parse_args()
    if "xgb" in args.models and not HAS_XGB:
        raise SystemExit("xgboost not installed")
    main(args)
