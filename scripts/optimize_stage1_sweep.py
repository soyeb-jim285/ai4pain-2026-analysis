"""
Stage-1 optimization sweep.

Grid over {model, feature_set, norm, calibration, anchor}. Reports per-subject
LOSO accuracy mean +/- std and validation accuracy mean +/- std.

Metric is balanced-accuracy-at-optimal-threshold equivalent via exact-12
NoPain decoder (matches real pipeline). Uses existing stage-1 helpers.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm


def _log(msg: str) -> None:
    print(msg, flush=True)

from src.final_pipeline import (
    HAS_XGB,
    ModelSpec,
    build_norm_map,
    exact12_binary_predictions,
    fit_binary_calibrator,
    fit_binary_proba,
    load_clean_features,
    metrics_binary,
    pain_binary,
    stage1_anchor_scores,
    stage1_feature_sets,
)


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


def run_config(df_all: pd.DataFrame, feat_cols: list[str], norm_map: dict,
                cfg: dict, cfg_idx: int, n_total: int) -> dict:
    t_cfg = time.time()
    _log(f"\n[cfg {cfg_idx+1}/{n_total}] starting  model={cfg['model']:6s} "
         f"fs={cfg['feature_set']:18s} norm={cfg['norm']:15s} "
         f"cal={cfg['calibration']:8s} anchor={cfg['anchor']:6s}")

    df_norm = norm_map[cfg["norm"]]
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feature_sets = stage1_feature_sets(train, feat_cols)
    selected = feature_sets[cfg["feature_set"]]
    if not selected:
        _log(f"[cfg {cfg_idx+1}/{n_total}] SKIP empty feature set")
        return {"status": "empty_feature_set", **cfg}
    spec = build_spec(cfg["model"])
    scaler = cfg["scaler"]
    _log(f"[cfg {cfg_idx+1}/{n_total}] n_features={len(selected)}  "
         f"train_rows={len(train)}  val_rows={len(val)}")

    train_probs = np.zeros((len(train), 2), dtype=np.float32)
    subjects = sorted(train["subject"].unique())
    n_subj = len(subjects)
    t_loso = time.time()
    fold_bar = tqdm(subjects, desc=f"  LOSO cfg{cfg_idx+1}/{n_total}",
                      ncols=90, leave=True, file=sys.stdout, mininterval=0.1,
                      miniters=1, dynamic_ncols=True)
    fold_times = []
    for fi, subj in enumerate(fold_bar):
        tr = train[train["subject"] != subj].reset_index(drop=True)
        te = train[train["subject"] == subj].reset_index(drop=True)
        t0 = time.time()
        # Text log per fold so user sees progress even if tqdm lags
        _log(f"    [cfg {cfg_idx+1}/{n_total}] fold {fi+1:2d}/{n_subj} "
             f"subj={int(subj):3d} n_train={len(tr)} fitting...")
        p = fit_binary_proba(tr, te, selected, pain_binary(tr["class"]),
                              scaler_name=scaler, spec=spec)
        train_probs[train["subject"] == subj] = p
        dt = time.time() - t0
        fold_times.append(dt)
        avg = sum(fold_times) / len(fold_times)
        eta = avg * (n_subj - fi - 1)
        _log(f"    [cfg {cfg_idx+1}/{n_total}] fold {fi+1:2d}/{n_subj} "
             f"subj={int(subj):3d} done in {dt:.2f}s  avg={avg:.2f}s/fold  "
             f"fold_eta={eta:.1f}s")
        fold_bar.set_postfix(last=f"{dt:.1f}s", avg=f"{avg:.1f}s")
    fold_bar.close()
    sys.stdout.flush()
    loso_s = time.time() - t_loso
    _log(f"[cfg {cfg_idx+1}/{n_total}] LOSO done in {loso_s:.1f}s "
         f"({loso_s/n_subj:.2f}s/fold)")

    t_val = time.time()
    val_probs = fit_binary_proba(train, val, selected,
                                  pain_binary(train["class"]),
                                  scaler_name=scaler, spec=spec)
    _log(f"[cfg {cfg_idx+1}/{n_total}] val fit in {time.time()-t_val:.1f}s")

    is_nopain_tr = 1 - pain_binary(train["class"])
    cal = fit_binary_calibrator(cfg["calibration"], train_probs[:, 0], is_nopain_tr)
    train_s = cal(train_probs[:, 0])
    val_s = cal(val_probs[:, 0])

    if cfg["anchor"] != "none":
        train_s = stage1_anchor_scores(train, df_norm, selected, train_s,
                                         mode=cfg["anchor"], lam=cfg["anchor_lambda"])
        val_s = stage1_anchor_scores(val, df_norm, selected, val_s,
                                       mode=cfg["anchor"], lam=cfg["anchor_lambda"])

    # Decode to 12 NoPain per subject (exact), compute per-subject accuracy
    def per_subj_acc(df_sub: pd.DataFrame, scores: np.ndarray) -> tuple[float, float, list]:
        y_true = pain_binary(df_sub["class"])
        y_pred = exact12_binary_predictions(df_sub, scores)
        rows = []
        for s in sorted(df_sub["subject"].unique()):
            m = (df_sub["subject"] == s).to_numpy()
            mm = metrics_binary(y_true[m], y_pred[m])
            rows.append(mm["accuracy"])
        return float(np.mean(rows)), float(np.std(rows)), rows

    loso_mean, loso_std, loso_list = per_subj_acc(train, train_s)
    val_mean, val_std, val_list = per_subj_acc(val, val_s)

    # AUC: train_s is p(NoPain); score NoPain as positive class
    from sklearn.metrics import roc_auc_score
    try:
        loso_auc = float(roc_auc_score(1 - pain_binary(train["class"]), train_s))
    except Exception:
        loso_auc = float("nan")
    try:
        val_auc = float(roc_auc_score(1 - pain_binary(val["class"]), val_s))
    except Exception:
        val_auc = float("nan")

    return {
        **cfg,
        "n_features": len(selected),
        "loso_acc_mean": loso_mean,
        "loso_acc_std": loso_std,
        "loso_acc_min": float(np.min(loso_list)) if loso_list else 0.0,
        "val_acc_mean": val_mean,
        "val_acc_std": val_std,
        "val_acc_min": float(np.min(val_list)) if val_list else 0.0,
        "loso_auc": loso_auc,
        "val_auc": val_auc,
    }


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all, feat_cols = load_clean_features(args.feature_parquet)
    norm_map = build_norm_map(df_all, feat_cols)

    models = args.models
    feature_sets = args.feature_sets
    norms = args.norms
    calibrations = args.calibrations
    anchors = args.anchors
    scaler = args.scaler

    configs = [
        {"model": m, "feature_set": fs, "norm": n, "calibration": c,
         "anchor": a, "anchor_lambda": args.anchor_lambda, "scaler": scaler}
        for m, fs, n, c, a in itertools.product(models, feature_sets, norms, calibrations, anchors)
    ]
    _log(f"[sweep] {len(configs)} configs  rows={len(df_all)} feats={len(feat_cols)}")
    t_start = time.time()
    rows = []
    partial_path = out_dir / "partial.csv"
    for idx, cfg in enumerate(configs):
        try:
            r = run_config(df_all, feat_cols, norm_map, cfg, idx, len(configs))
            rows.append(r)
            elapsed = time.time() - t_start
            remain_est = elapsed / (idx + 1) * (len(configs) - idx - 1)
            _log(f"[cfg {idx+1}/{len(configs)}] RESULT  "
                 f"loso_acc={r['loso_acc_mean']:.4f}+-{r['loso_acc_std']:.3f}  "
                 f"val_acc={r['val_acc_mean']:.4f}+-{r['val_acc_std']:.3f}  "
                 f"auc_loso={r['loso_auc']:.3f} auc_val={r['val_auc']:.3f}  "
                 f"elapsed={elapsed:.1f}s  eta={remain_est:.1f}s")
            # Write partial after each config so user can peek while running
            pd.DataFrame(rows).to_csv(partial_path, index=False)
        except Exception as e:
            _log(f"[cfg {idx+1}/{len(configs)}] FAIL {cfg}: {e}")
            import traceback
            traceback.print_exc()

    summary = pd.DataFrame(rows).sort_values("loso_acc_mean", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    _log(f"\n[done] total time: {time.time()-t_start:.1f}s")

    top5 = summary.head(5)[
        ["model", "feature_set", "norm", "calibration", "anchor",
         "n_features", "loso_acc_mean", "loso_acc_std",
         "val_acc_mean", "val_acc_std", "loso_auc", "val_auc"]
    ]

    print("\n=== TOP 5 by LOSO acc ===")
    print(top5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    (out_dir / "report.md").write_text("\n".join([
        "# Stage 1 Optimization Sweep",
        "",
        f"- configs: {len(configs)}",
        f"- scaler: `{scaler}`  anchor_lambda: {args.anchor_lambda}",
        "",
        "## Top 10 by LOSO acc",
        "",
        summary.head(10).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## All (sorted)",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
    ]))
    print(f"\n[done] wrote {out_dir}/summary.csv and report.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature-parquet", default="results/tables/all_features_merged_1022.parquet")
    p.add_argument("--models", nargs="+", default=["logreg", "xgb"],
                    choices=["logreg", "xgb", "rf"])
    p.add_argument("--feature-sets", nargs="+",
                    default=["bvp_eda_core", "eda_bvp_fwd8", "bvp_eda_resp_small"],
                    choices=["bvp_only", "eda_only", "bvp_eda_core",
                             "bvp_eda_resp_small", "eda_bvp_fwd8", "all"])
    p.add_argument("--norms", nargs="+", default=["subject_robust", "subject_z"],
                    choices=["global", "subject_z", "subject_robust"])
    p.add_argument("--calibrations", nargs="+", default=["sigmoid", "isotonic"],
                    choices=["none", "sigmoid", "isotonic"])
    p.add_argument("--anchors", nargs="+", default=["none", "center"],
                    choices=["none", "center", "z"])
    p.add_argument("--anchor-lambda", type=float, default=0.5)
    p.add_argument("--scaler", default="std", choices=["std", "robust"])
    p.add_argument("--output-dir", default="results/final/stage1_sweep")
    args = p.parse_args()
    if "xgb" in args.models and not HAS_XGB:
        raise SystemExit("xgboost not installed")
    main(args)
