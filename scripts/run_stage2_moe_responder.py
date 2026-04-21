"""
Mixture-of-Experts stage-2 by responder-strength cluster.

Hypothesis: weak responders (small NoPain->Pain amplitude delta) have
arm-vs-hand signal buried in noise; a dedicated expert may learn their
low-amplitude pattern better than a shared model.

Approach:
  1. For each subject, compute responder_strength = ||mean(Pain) - mean(NoPain)||_2
     on globally z-scored features (train subjects only for fit).
  2. Cluster train subjects into K strata by log-responder-strength quantiles.
  3. For each held-out test subject, assign stratum via its own strength score.
  4. Fit one stage-2 logreg per stratum (expert) on pain rows of that stratum.
  5. Predict held-out subject's pain rows with the expert matching its stratum.
  6. LOSO + val evaluation with exact-12 arm-vs-hand decoding.

Baseline compared: shared-expert (identical data, k=1).
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
    ModelSpec,
    armhand_binary,
    fit_binary_calibrator,
    fit_binary_proba,
    load_clean_features,
    metrics_binary,
    stage2_feature_sets,
)


def compute_responder_strength(df: pd.DataFrame, feat_cols: list[str], ref_subjects: list[int]) -> pd.Series:
    """Globally z-score features using ref_subjects' rows, then compute per-subject L2 delta."""
    ref_rows = df[df["subject"].isin(ref_subjects)]
    mu = ref_rows[feat_cols].mean(axis=0)
    sd = ref_rows[feat_cols].std(axis=0, ddof=0).replace(0.0, 1.0)
    zed = (df[feat_cols] - mu) / sd
    zed = zed.fillna(0.0)
    tmp = df[["subject", "class"]].copy()
    tmp = pd.concat([tmp, zed], axis=1)
    rows = {}
    for subj, sub in tmp.groupby("subject"):
        pain = sub[sub["class"].isin(ARM_HAND)][feat_cols].mean(axis=0)
        nopain = sub[sub["class"] == "NoPain"][feat_cols].mean(axis=0)
        if pain.isna().any() or nopain.isna().any():
            rows[int(subj)] = 0.0
            continue
        rows[int(subj)] = float(np.linalg.norm((pain - nopain).to_numpy()))
    return pd.Series(rows, name="responder_strength")


def assign_stratum(strength: pd.Series, train_subjects: list[int], k: int) -> tuple[pd.Series, np.ndarray]:
    """Assign each subject to one of K strata by log-strength quantile (cut on train).
    Returns (subject -> stratum, strata_edges)."""
    train_strength = strength.loc[train_subjects].copy()
    log_s = np.log1p(train_strength)
    edges = np.quantile(log_s, np.linspace(0, 1, k + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    stratum = pd.Series(np.zeros(len(strength), dtype=int), index=strength.index)
    for i, s in enumerate(strength.index):
        v = np.log1p(strength.loc[s])
        b = int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, k - 1))
        stratum.loc[s] = b
    return stratum, edges


def build_spec(model: str) -> ModelSpec:
    if model == "logreg":
        return ModelSpec("logreg", {"C": 1.0})
    if model == "xgb":
        return ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08,
                                  "subsample": 1.0, "colsample_bytree": 1.0})
    raise ValueError(model)


def exact12_arm_predictions(df_sub: pd.DataFrame, arm_scores: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-arm_scores[mask])]
        arm_idx = set(order[:12].tolist())
        pred[mask] = [0 if i in arm_idx else 1 for i in idx]
    return pred


def subject_z_norm(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    mu = out.groupby("subject")[feat_cols].transform("mean")
    sd = out.groupby("subject")[feat_cols].transform("std", ddof=0)
    sd = sd.where(sd > 0, 1.0)
    out[feat_cols] = ((out[feat_cols] - mu) / sd).fillna(0.0).astype(np.float32)
    return out


def run_moe(df: pd.DataFrame, feat_cols: list[str], feature_set: str, model: str,
             scaler: str, calibration: str, k: int, strength: pd.Series) -> dict:
    df_norm = subject_z_norm(df, feat_cols)
    train_df = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val_df = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    fs_all = stage2_feature_sets(train_df, feat_cols)
    selected = fs_all[feature_set]

    train_pain = train_df[train_df["class"].isin(ARM_HAND)].reset_index(drop=True)
    val_pain = val_df[val_df["class"].isin(ARM_HAND)].reset_index(drop=True)

    train_subjects = sorted(train_pain["subject"].unique().tolist())
    stratum, edges = assign_stratum(strength, train_subjects, k)
    stratum_counts = stratum.loc[train_subjects].value_counts().sort_index()

    # LOSO: for each held-out subject, assign to stratum, train expert on remaining train subjects in same stratum
    spec = build_spec(model)
    train_probs = np.zeros((len(train_pain), 2), dtype=np.float32)
    for held in train_subjects:
        stratum_of_held = int(stratum.loc[held])
        expert_subjects = [s for s in train_subjects if s != held and int(stratum.loc[s]) == stratum_of_held]
        if len(expert_subjects) < 5:
            # fallback: use all remaining train subjects
            expert_subjects = [s for s in train_subjects if s != held]
        tr = train_pain[train_pain["subject"].isin(expert_subjects)].reset_index(drop=True)
        te = train_pain[train_pain["subject"] == held].reset_index(drop=True)
        probs = fit_binary_proba(tr, te, selected, armhand_binary(tr["class"]),
                                  scaler_name=scaler, spec=spec)
        train_probs[train_pain["subject"] == held] = probs

    # Validation: train one expert per stratum on all train subjects in that stratum,
    # predict each val subject with the expert for its assigned stratum.
    val_probs = np.zeros((len(val_pain), 2), dtype=np.float32)
    for val_subj in sorted(val_pain["subject"].unique()):
        stratum_of_val = int(stratum.loc[val_subj])
        expert_subjects = [s for s in train_subjects if int(stratum.loc[s]) == stratum_of_val]
        if len(expert_subjects) < 5:
            expert_subjects = train_subjects
        tr = train_pain[train_pain["subject"].isin(expert_subjects)].reset_index(drop=True)
        te = val_pain[val_pain["subject"] == val_subj].reset_index(drop=True)
        probs = fit_binary_proba(tr, te, selected, armhand_binary(tr["class"]),
                                  scaler_name=scaler, spec=spec)
        val_probs[val_pain["subject"] == val_subj] = probs

    cal = fit_binary_calibrator(calibration, train_probs[:, 0],
                                 (train_pain["class"] == "PainArm").astype(int).to_numpy())
    train_arm = cal(train_probs[:, 0])
    val_arm = cal(val_probs[:, 0])

    results = {}
    per_subj_rows = []
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
            per_subj_rows.append({
                "split": name, "subject": int(subj),
                "stratum": int(stratum.loc[subj]),
                "responder_strength": float(strength.loc[subj]),
                "macro_f1": mm["macro_f1"],
            })
        results[f"{name}_per_subject_std"] = float(np.std(sub_f1s))
        results[f"{name}_per_subject_min"] = float(np.min(sub_f1s))

    results["k"] = k
    results["stratum_counts"] = stratum_counts.to_dict()
    return {"summary": results, "per_subject": pd.DataFrame(per_subj_rows),
             "edges": edges.tolist(), "stratum": stratum}


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features(args.feature_parquet)
    train_subjects = sorted(df_all[df_all["split"] == "train"]["subject"].unique().tolist())
    print(f"[load] rows={len(df_all)} train_subj={len(train_subjects)} feats={len(feat_cols)}")

    strength = compute_responder_strength(df_all, feat_cols, train_subjects)
    print(f"[strength] train range=[{strength.loc[train_subjects].min():.1f}, "
          f"{strength.loc[train_subjects].max():.1f}] "
          f"median={strength.loc[train_subjects].median():.1f}")

    summary_rows = []
    per_subj_frames = []
    for k in args.k_values:
        print(f"\n[run] k={k} model={args.model}")
        res = run_moe(df_all, feat_cols, args.feature_set, args.model,
                       args.scaler, args.calibration, k, strength)
        s = res["summary"]
        print(f"    loso_macro_f1={s['loso_macro_f1']:.4f} "
              f"val_macro_f1={s['val_macro_f1']:.4f} "
              f"per_subj_std={s['loso_per_subject_std']:.3f} "
              f"per_subj_min={s['loso_per_subject_min']:.3f} "
              f"strata={s['stratum_counts']}")
        summary_rows.append({"k": k, **{kk: vv for kk, vv in s.items() if kk != "stratum_counts"},
                              "stratum_counts": str(s["stratum_counts"])})
        per_subj = res["per_subject"].copy()
        per_subj["k"] = k
        per_subj_frames.append(per_subj)

    summary = pd.DataFrame(summary_rows).sort_values("loso_macro_f1", ascending=False)
    per_subj_all = pd.concat(per_subj_frames, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_subj_all.to_csv(out_dir / "per_subject.csv", index=False)
    strength.to_csv(out_dir / "responder_strength.csv", header=True)

    report = ["# Stage 2 MoE-by-Responder-Cluster Sweep", "",
              f"- model: `{args.model}`  feature_set: `{args.feature_set}`",
              f"- scaler: `{args.scaler}`  calibration: `{args.calibration}`",
              f"- norm: `subject_z` (fixed)",
              "", "## Results (sorted by LOSO macro-F1)", "",
              summary.to_markdown(index=False, floatfmt=".4f")]
    (out_dir / "report.md").write_text("\n".join(report))
    print(f"\n[done] wrote {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature-parquet", default="results/tables/all_features_merged_1022.parquet")
    p.add_argument("--feature-set", default="resp_all")
    p.add_argument("--model", default="logreg", choices=["logreg", "xgb"])
    p.add_argument("--scaler", default="robust", choices=["std", "robust"])
    p.add_argument("--calibration", default="isotonic", choices=["none", "sigmoid", "isotonic"])
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--output-dir", default="results/final/stage2_moe_responder")
    main(p.parse_args())
