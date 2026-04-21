"""
Stage-2 optimization sweep (arm-vs-hand on pain-only rows).

Grid over {model, feature_set, norm, calibration, scaler, anchor}. Reports
per-subject LOSO accuracy mean +/- std and validation accuracy mean +/- std,
plus binary AUC (p(PainArm) vs is_arm label).

Decoder: exact-12 per subject (top-12 by p(Arm) labelled Arm). Matches
run_stage2.py pipeline. Calibrator label direction matches run_stage2.py:114.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND,
    HAS_XGB,
    ModelSpec,
    apply_subject_robust,
    apply_subject_z,
    armhand_binary,
    fit_binary_calibrator,
    fit_binary_proba,
    load_clean_features,
    metrics_binary,
    stage2_feature_sets,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def build_spec(model: str) -> ModelSpec:
    if model == "logreg":
        return ModelSpec("logreg", {"C": 1.0})
    if model == "xgb":
        return ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4,
                                  "learning_rate": 0.08, "subsample": 1.0,
                                  "colsample_bytree": 1.0})
    if model == "rf":
        return ModelSpec("rf", {"n_estimators": 400, "max_depth": None})
    raise ValueError(model)


def norm_df(df: pd.DataFrame, feat_cols: list[str], how: str) -> pd.DataFrame:
    if how == "global":
        return df.copy()
    if how == "subject_z":
        return apply_subject_z(df, feat_cols)
    if how == "subject_robust":
        return apply_subject_robust(df, feat_cols)
    raise ValueError(how)


def nopain_anchor(df: pd.DataFrame, feat_cols: list[str], mode: str) -> pd.DataFrame:
    """Per-subject norm using ONLY NoPain rows as baseline stats."""
    out = df.copy()
    for subj in sorted(df["subject"].unique()):
        mask = df["subject"] == subj
        base = df.loc[mask & (df["class"] == "NoPain"), feat_cols]
        if base.empty:
            continue
        if mode == "center":
            mu = base.mean(axis=0)
            out.loc[mask, feat_cols] = (df.loc[mask, feat_cols] - mu).astype(np.float32)
        elif mode == "z":
            mu = base.mean(axis=0)
            sd = base.std(axis=0, ddof=0)
            sd = sd.where(sd > 0, 1.0)
            out.loc[mask, feat_cols] = ((df.loc[mask, feat_cols] - mu) / sd).astype(np.float32)
        else:
            raise ValueError(mode)
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out


def exact12_arm_predictions(df_sub: pd.DataFrame, arm_scores: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-arm_scores[mask])]
        arm_idx = set(order[:12].tolist())
        pred[mask] = [0 if i in arm_idx else 1 for i in idx]
    return pred


def run_config(df_all: pd.DataFrame, feat_cols: list[str], cfg: dict,
                cfg_idx: int, n_total: int) -> dict:
    t_cfg = time.time()
    _log(f"\n[cfg {cfg_idx+1}/{n_total}] starting  model={cfg['model']:6s} "
         f"fs={cfg['feature_set']:16s} norm={cfg['norm']:15s} "
         f"cal={cfg['calibration']:8s} anchor={cfg['anchor']:6s} "
         f"scaler={cfg['scaler']:6s}")

    # Apply subject-level normalization
    df_norm = norm_df(df_all, feat_cols, cfg["norm"])
    # Apply NoPain anchor on top if requested
    if cfg["anchor"] != "none":
        df_norm = nopain_anchor(df_norm, feat_cols, cfg["anchor"])

    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    fs_map = stage2_feature_sets(train, feat_cols)
    selected = fs_map[cfg["feature_set"]]
    if not selected:
        _log(f"[cfg {cfg_idx+1}/{n_total}] SKIP empty feature set")
        return {"status": "empty_feature_set", **cfg}

    spec = build_spec(cfg["model"])
    train_pain = train[train["class"].isin(ARM_HAND)].reset_index(drop=True)
    val_pain = val[val["class"].isin(ARM_HAND)].reset_index(drop=True)
    _log(f"[cfg {cfg_idx+1}/{n_total}] n_features={len(selected)}  "
         f"train_pain_rows={len(train_pain)}  val_pain_rows={len(val_pain)}")

    subjects = sorted(train_pain["subject"].unique())
    n_subj = len(subjects)
    train_probs = np.zeros((len(train_pain), 2), dtype=np.float32)
    t_loso = time.time()
    fold_times = []
    for fi, subj in enumerate(subjects):
        tr = train_pain[train_pain["subject"] != subj].reset_index(drop=True)
        te = train_pain[train_pain["subject"] == subj].reset_index(drop=True)
        t0 = time.time()
        p = fit_binary_proba(tr, te, selected, armhand_binary(tr["class"]),
                              scaler_name=cfg["scaler"], spec=spec)
        train_probs[train_pain["subject"] == subj] = p
        dt = time.time() - t0
        fold_times.append(dt)
        if fi == 0 or (fi + 1) % 10 == 0 or fi + 1 == n_subj:
            avg = sum(fold_times) / len(fold_times)
            eta = avg * (n_subj - fi - 1)
            _log(f"    [cfg {cfg_idx+1}/{n_total}] fold {fi+1:2d}/{n_subj} "
                 f"subj={int(subj):3d} last={dt:.2f}s avg={avg:.2f}s eta={eta:.0f}s")
    loso_s = time.time() - t_loso
    _log(f"[cfg {cfg_idx+1}/{n_total}] LOSO done in {loso_s:.1f}s "
         f"({loso_s/n_subj:.2f}s/fold)")

    t_val = time.time()
    val_probs = fit_binary_proba(train_pain, val_pain, selected,
                                  armhand_binary(train_pain["class"]),
                                  scaler_name=cfg["scaler"], spec=spec)
    _log(f"[cfg {cfg_idx+1}/{n_total}] val fit in {time.time()-t_val:.1f}s")

    # Calibrator: input p(PainArm), labels is_arm=1 (matches run_stage2.py:114)
    is_arm_tr = (train_pain["class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator(cfg["calibration"], train_probs[:, 0], is_arm_tr)
    train_s = cal(train_probs[:, 0])
    val_s = cal(val_probs[:, 0])

    def per_subj_metric(df_sub: pd.DataFrame, scores: np.ndarray):
        y_true = (df_sub["class"] == "PainHand").astype(int).to_numpy()
        y_pred = exact12_arm_predictions(df_sub, scores)
        rows = []
        for s in sorted(df_sub["subject"].unique()):
            m = (df_sub["subject"] == s).to_numpy()
            mm = metrics_binary(y_true[m], y_pred[m])
            rows.append(mm["accuracy"])
        return float(np.mean(rows)), float(np.std(rows)), float(np.min(rows)), rows

    loso_mean, loso_std, loso_min, loso_list = per_subj_metric(train_pain, train_s)
    val_mean, val_std, val_min, val_list = per_subj_metric(val_pain, val_s)

    from sklearn.metrics import roc_auc_score
    try:
        loso_auc = float(roc_auc_score(is_arm_tr, train_s))
    except Exception:
        loso_auc = float("nan")
    try:
        val_is_arm = (val_pain["class"] == "PainArm").astype(int).to_numpy()
        val_auc = float(roc_auc_score(val_is_arm, val_s))
    except Exception:
        val_auc = float("nan")

    _log(f"[cfg {cfg_idx+1}/{n_total}] cfg_time={time.time()-t_cfg:.1f}s")
    return {
        **cfg,
        "n_features": len(selected),
        "loso_acc_mean": loso_mean,
        "loso_acc_std": loso_std,
        "loso_acc_min": loso_min,
        "val_acc_mean": val_mean,
        "val_acc_std": val_std,
        "val_acc_min": val_min,
        "loso_auc": loso_auc,
        "val_auc": val_auc,
    }


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features(args.feature_parquet)

    configs = [
        {"model": m, "feature_set": fs, "norm": n, "calibration": c,
         "anchor": a, "scaler": s}
        for m, fs, n, c, a, s in itertools.product(
            args.models, args.feature_sets, args.norms,
            args.calibrations, args.anchors, args.scalers)
    ]
    _log(f"[sweep] {len(configs)} configs  rows={len(df_all)} feats={len(feat_cols)}")

    t_start = time.time()
    rows = []
    partial_path = out_dir / "partial.csv"
    for idx, cfg in enumerate(configs):
        try:
            r = run_config(df_all, feat_cols, cfg, idx, len(configs))
            rows.append(r)
            elapsed = time.time() - t_start
            remain = elapsed / (idx + 1) * (len(configs) - idx - 1)
            _log(f"[cfg {idx+1}/{len(configs)}] RESULT  "
                 f"loso_acc={r.get('loso_acc_mean',float('nan')):.4f}+-{r.get('loso_acc_std',0):.3f} "
                 f"val_acc={r.get('val_acc_mean',float('nan')):.4f}+-{r.get('val_acc_std',0):.3f} "
                 f"loso_auc={r.get('loso_auc',float('nan')):.3f} val_auc={r.get('val_auc',float('nan')):.3f} "
                 f"elapsed={elapsed:.0f}s eta={remain:.0f}s")
            pd.DataFrame(rows).to_csv(partial_path, index=False)
        except Exception as e:
            _log(f"[cfg {idx+1}/{len(configs)}] FAIL {cfg}: {e}")
            import traceback
            traceback.print_exc()

    summary = pd.DataFrame(rows).sort_values("loso_acc_mean", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    _log(f"\n[done] total time: {time.time()-t_start:.1f}s")

    print("\n=== TOP 10 by LOSO acc ===", flush=True)
    cols = ["model", "feature_set", "norm", "calibration", "anchor", "scaler",
            "n_features", "loso_acc_mean", "loso_acc_std",
            "val_acc_mean", "val_acc_std", "loso_auc", "val_auc"]
    print(summary.head(10)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    (out_dir / "report.md").write_text("\n".join([
        "# Stage 2 Optimization Sweep",
        "",
        f"- configs: {len(configs)}",
        "",
        "## Top 10 by LOSO acc",
        "",
        summary.head(10).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## All (sorted)",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
    ]))
    _log(f"[done] wrote {out_dir}/summary.csv and report.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature-parquet", default="results/tables/all_features_merged_1022.parquet")
    p.add_argument("--models", nargs="+", default=["logreg", "xgb"],
                    choices=["logreg", "xgb", "rf"])
    p.add_argument("--feature-sets", nargs="+",
                    default=["resp_all", "resp_top20", "eda_resp_top30",
                             "bvp_resp_top30", "all_top40"],
                    choices=["resp_all", "resp_top20", "resp_bvp5",
                             "eda_resp_top30", "bvp_resp_top30", "all_top40"])
    p.add_argument("--norms", nargs="+", default=["subject_z", "subject_robust"],
                    choices=["global", "subject_z", "subject_robust"])
    p.add_argument("--calibrations", nargs="+", default=["sigmoid", "isotonic"],
                    choices=["none", "sigmoid", "isotonic"])
    p.add_argument("--anchors", nargs="+", default=["none"],
                    choices=["none", "center", "z"])
    p.add_argument("--scalers", nargs="+", default=["robust"],
                    choices=["std", "robust"])
    p.add_argument("--output-dir", default="results/final/stage2_sweep")
    args = p.parse_args()
    if "xgb" in args.models and not HAS_XGB:
        raise SystemExit("xgboost not installed")
    main(args)
