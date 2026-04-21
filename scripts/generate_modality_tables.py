from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.final_pipeline import (
    ARM_HAND,
    CLASS_ORDER_3,
    ModelSpec,
    armhand_binary,
    build_norm_map,
    class_codes_3,
    exact_count_decode,
    fit_binary_calibrator,
    fit_binary_proba,
    load_clean_features,
    metrics_multiclass,
    stage1_anchor_scores,
)


MODALITY_MAP = {
    "bvp": ["bvp"],
    "eda": ["eda"],
    "resp": ["resp"],
    "spo2": ["spo2"],
    "bvp_eda": ["bvp", "eda"],
    "all": ["bvp", "eda", "resp", "spo2"],
}


BASELINE_CONFIG = {
    "stage1_model": "xgb",
    "stage1_norm": "subject_robust",
    "stage1_scaler": "std",
    "stage1_calibration": "sigmoid",
    "stage1_anchor_mode": "center",
    "stage1_anchor_lambda": 0.5,
    "stage1_xgb": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0},
    "stage1_logreg_c": 1.0,
    "stage2_model": "logreg",
    "stage2_norm": "subject_z",
    "stage2_scaler": "robust",
    "stage2_calibration": "isotonic",
    "stage2_anchor_mode": "none",
    "stage2_xgb": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0},
    "stage2_logreg_c": 1.0,
    "w0": 0.8, "w1": 1.2, "w2": 1.4,
}


def channel_of(name: str) -> str:
    n = name.lower()
    for tag in ("bvp", "eda", "resp", "spo2", "spo_2"):
        if tag in n:
            return "spo2" if tag == "spo_2" else tag
    return "other"


def default_feature_parquet(resample_tag: str) -> Path:
    return Path(f"results/tables/all_features_merged_{resample_tag}.parquet")


def stage1_spec_from(cfg: dict) -> ModelSpec:
    if cfg["stage1_model"] == "xgb":
        return ModelSpec("xgb", dict(cfg["stage1_xgb"]))
    if cfg["stage1_model"] == "logreg":
        return ModelSpec("logreg", {"C": cfg.get("stage1_logreg_c", 1.0)})
    raise ValueError(cfg["stage1_model"])


def stage2_spec_from(cfg: dict) -> ModelSpec:
    if cfg["stage2_model"] == "xgb":
        return ModelSpec("xgb", dict(cfg["stage2_xgb"]))
    if cfg["stage2_model"] == "logreg":
        return ModelSpec("logreg", {"C": cfg.get("stage2_logreg_c", 1.0)})
    raise ValueError(cfg["stage2_model"])


def predicted_anchor_map(df_sub: pd.DataFrame, score_nopain: np.ndarray) -> dict[int, list[str]]:
    out = {}
    for subject in sorted(df_sub["subject"].unique()):
        sub = df_sub[df_sub["subject"] == subject].reset_index(drop=True)
        p = score_nopain[df_sub["subject"] == subject]
        order = np.argsort(-p)[:12]
        out[int(subject)] = sub.iloc[order]["segment_id"].tolist()
    return out


def adapt_with_anchor(df_norm, df_global, feat_cols, anchor_map, mode):
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


def fit_stage1(df_norm, feat_cols, spec, scaler_name, calibration, anchor_mode, anchor_lambda, df_global):
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


def fit_stage2(df_norm, df_global, feat_cols, spec, scaler_name, calibration, anchor_mode, train_stage1_scores, val_stage1_scores):
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


def decode_joint_per_subject(df_sub, stage1_probs, stage2_probs, w0, w1, w2):
    y_pred = np.zeros(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        p_pain = np.clip(np.nan_to_num(stage1_probs[mask, 1], nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6), 1e-6, 1.0 - 1e-6)
        p_arm = np.clip(np.nan_to_num(stage2_probs[mask, 0], nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6), 1e-6, 1.0 - 1e-6)
        log_scores = np.column_stack([
            w0 * np.log1p(-p_pain),
            w1 * np.log(p_pain) + w2 * np.log(p_arm),
            w1 * np.log(p_pain) + w2 * np.log1p(-p_arm),
        ])
        log_scores = np.nan_to_num(log_scores, nan=-1e9, neginf=-1e9, posinf=1e9)
        y_pred[mask] = exact_count_decode(log_scores, [12, 12, 12])
    return y_pred


def score_from_probs(df_sub, stage1_probs, stage2_probs, w0, w1, w2):
    y_pred = decode_joint_per_subject(df_sub, stage1_probs, stage2_probs, w0, w1, w2)
    y_true = class_codes_3(df_sub["class"])
    met = metrics_multiclass(y_true, y_pred)
    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_ORDER_3))), average=None, zero_division=0
    )
    class_metrics = []
    for cls_idx, cls_name in enumerate(CLASS_ORDER_3):
        class_metrics.append({
            "class": cls_name,
            "precision": float(prec_arr[cls_idx]),
            "recall": float(rec_arr[cls_idx]),
            "f1": float(f1_arr[cls_idx]),
            "support": int(sup_arr[cls_idx]),
        })
    return met, class_metrics, y_pred


def stage1_key(cfg: dict) -> tuple:
    return (
        cfg["stage1_model"], cfg["stage1_norm"], cfg["stage1_scaler"], cfg["stage1_calibration"],
        cfg["stage1_anchor_mode"], round(float(cfg["stage1_anchor_lambda"]), 4),
        json.dumps(cfg.get("stage1_xgb", {}), sort_keys=True),
        round(float(cfg.get("stage1_logreg_c", 1.0)), 4),
    )


def stage2_key(cfg: dict) -> tuple:
    return (
        cfg["stage2_model"], cfg["stage2_norm"], cfg["stage2_scaler"], cfg["stage2_calibration"],
        cfg["stage2_anchor_mode"],
        json.dumps(cfg.get("stage2_xgb", {}), sort_keys=True),
        round(float(cfg.get("stage2_logreg_c", 1.0)), 4),
    )


def compute_stage1(cfg, df_all, norm_map, feat_cols):
    spec = stage1_spec_from(cfg)
    return fit_stage1(
        norm_map[cfg["stage1_norm"]], feat_cols, spec,
        scaler_name=cfg["stage1_scaler"],
        calibration=cfg["stage1_calibration"],
        anchor_mode=cfg["stage1_anchor_mode"],
        anchor_lambda=cfg["stage1_anchor_lambda"],
        df_global=df_all,
    )


def compute_stage2(cfg, df_all, norm_map, feat_cols, train_s1_scores, val_s1_scores):
    spec = stage2_spec_from(cfg)
    return fit_stage2(
        norm_map[cfg["stage2_norm"]], df_all, feat_cols, spec,
        scaler_name=cfg["stage2_scaler"],
        calibration=cfg["stage2_calibration"],
        anchor_mode=cfg["stage2_anchor_mode"],
        train_stage1_scores=train_s1_scores,
        val_stage1_scores=val_s1_scores,
    )


def evaluate_config(df_all, norm_map, feat_cols, cfg, s1_cache=None, s2_cache=None):
    s1_k = stage1_key(cfg)
    if s1_cache is not None and s1_k in s1_cache:
        train_s1, val_s1, s1_train_probs, s1_val_probs = s1_cache[s1_k]
    else:
        train_s1, val_s1, s1_train_probs, s1_val_probs = compute_stage1(cfg, df_all, norm_map, feat_cols)
        if s1_cache is not None:
            s1_cache[s1_k] = (train_s1, val_s1, s1_train_probs, s1_val_probs)

    s2_k = stage2_key(cfg)
    if s2_cache is not None and s2_k in s2_cache:
        train_s2, val_s2, s2_train_probs, s2_val_probs = s2_cache[s2_k]
    else:
        train_s2, val_s2, s2_train_probs, s2_val_probs = compute_stage2(
            cfg, df_all, norm_map, feat_cols, s1_train_probs[:, 0], s1_val_probs[:, 0]
        )
        if s2_cache is not None:
            s2_cache[s2_k] = (train_s2, val_s2, s2_train_probs, s2_val_probs)

    train_overall, train_classwise, _ = score_from_probs(train_s1, s1_train_probs, s2_train_probs, cfg["w0"], cfg["w1"], cfg["w2"])
    val_overall, val_classwise, _ = score_from_probs(val_s1, s1_val_probs, s2_val_probs, cfg["w0"], cfg["w1"], cfg["w2"])
    return {
        "train_loso": {"overall": train_overall, "classwise": train_classwise},
        "validation": {"overall": val_overall, "classwise": val_classwise},
    }


def medium_grid() -> list[dict]:
    stage1_variants = []
    for model in ["xgb", "logreg"]:
        for norm in ["subject_z", "subject_robust"]:
            for anchor in ["none", "center"]:
                stage1_variants.append({"stage1_model": model, "stage1_norm": norm, "stage1_anchor_mode": anchor})
    stage2_variants = []
    for model in ["xgb", "logreg"]:
        for norm in ["subject_z", "subject_robust"]:
            stage2_variants.append({"stage2_model": model, "stage2_norm": norm})
    weight_variants = [
        {"w0": 0.8, "w1": 1.2, "w2": 1.4},
        {"w0": 1.0, "w1": 1.0, "w2": 1.0},
        {"w0": 0.6, "w1": 1.4, "w2": 1.6},
    ]
    configs = []
    for s1 in stage1_variants:
        for s2 in stage2_variants:
            for w in weight_variants:
                cfg = dict(BASELINE_CONFIG)
                cfg.update(s1)
                cfg.update(s2)
                cfg.update(w)
                configs.append(cfg)
    return configs


def optimize_modality(df_all, norm_map, feat_cols, modality_name):
    configs = medium_grid()
    print(f"  [{modality_name}] sweeping {len(configs)} configs")
    s1_cache: dict = {}
    s2_cache_by_s1: dict = {}
    results = []
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        s1_k = stage1_key(cfg)
        s2_cache = s2_cache_by_s1.setdefault(s1_k, {})
        res = evaluate_config(df_all, norm_map, feat_cols, cfg, s1_cache=s1_cache, s2_cache=s2_cache)
        row = {
            "stage1_model": cfg["stage1_model"],
            "stage1_norm": cfg["stage1_norm"],
            "stage1_anchor_mode": cfg["stage1_anchor_mode"],
            "stage2_model": cfg["stage2_model"],
            "stage2_norm": cfg["stage2_norm"],
            "w0": cfg["w0"], "w1": cfg["w1"], "w2": cfg["w2"],
            "train_loso_macro_f1": res["train_loso"]["overall"]["macro_f1"],
            "val_macro_f1": res["validation"]["overall"]["macro_f1"],
            "train_loso_accuracy": res["train_loso"]["overall"]["accuracy"],
            "val_accuracy": res["validation"]["overall"]["accuracy"],
        }
        results.append((cfg, res, row))
        if i % 8 == 0:
            print(f"    progress {i}/{len(configs)} elapsed={time.time() - t0:.1f}s")
    best_idx = max(range(len(results)), key=lambda i: results[i][2]["train_loso_macro_f1"])
    best_cfg, best_res, _ = results[best_idx]
    print(f"  [{modality_name}] best train_loso_macro_f1={results[best_idx][2]['train_loso_macro_f1']:.4f}, val={results[best_idx][2]['val_macro_f1']:.4f} at idx {best_idx}")
    grid_df = pd.DataFrame([r[2] for r in results])
    return best_cfg, best_res, grid_df


def config_summary(cfg: dict) -> str:
    return (
        f"stage1={cfg['stage1_model']}/{cfg['stage1_norm']}/anchor={cfg['stage1_anchor_mode']}, "
        f"stage2={cfg['stage2_model']}/{cfg['stage2_norm']}/anchor={cfg['stage2_anchor_mode']}, "
        f"w=({cfg['w0']},{cfg['w1']},{cfg['w2']})"
    )


def main(args: argparse.Namespace) -> None:
    feature_fp = args.feature_parquet or default_feature_parquet(args.resample_tag)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all, feat_cols = load_clean_features(feature_fp)
    norm_map = build_norm_map(df_all, feat_cols)

    overall_rows = []
    class_rows = []
    grid_rows = []
    best_config_rows = []
    for name, tags in MODALITY_MAP.items():
        cols = [c for c in feat_cols if channel_of(c) in tags]
        if not cols:
            print(f"[skip] {name}: no features")
            continue
        print(f"[run ] {name}: {len(cols)} features")
        if args.optimize:
            best_cfg, best_res, grid_df = optimize_modality(df_all, norm_map, cols, name)
            grid_df["modality"] = name
            grid_rows.append(grid_df)
            best_config_rows.append({"modality": name, "n_features": len(cols), **{k: v for k, v in best_cfg.items() if not isinstance(v, dict)}, "selection_train_loso_macro_f1": best_res["train_loso"]["overall"]["macro_f1"], "val_macro_f1": best_res["validation"]["overall"]["macro_f1"]})
            res = best_res
            print(f"       best: {config_summary(best_cfg)}")
        else:
            res = evaluate_config(df_all, norm_map, cols, BASELINE_CONFIG)
            best_config_rows.append({"modality": name, "n_features": len(cols), **{k: v for k, v in BASELINE_CONFIG.items() if not isinstance(v, dict)}, "val_macro_f1": res["validation"]["overall"]["macro_f1"]})
        for split_name in ("train_loso", "validation"):
            overall_rows.append({"split": split_name, "modality": name, "n_features": len(cols), **res[split_name]["overall"]})
            for cls_row in res[split_name]["classwise"]:
                class_rows.append({"split": split_name, "modality": name, **cls_row})

    overall_df = pd.DataFrame(overall_rows)
    class_df = pd.DataFrame(class_rows)
    overall_df.to_csv(out_dir / "modality_overall.csv", index=False)
    class_df.to_csv(out_dir / "modality_classwise.csv", index=False)
    if best_config_rows:
        pd.DataFrame(best_config_rows).to_csv(out_dir / "best_config_per_modality.csv", index=False)
    if grid_rows:
        full_grid = pd.concat(grid_rows, ignore_index=True)
        full_grid.to_csv(out_dir / "grid_search.csv", index=False)

    val_overall = overall_df[overall_df["split"] == "validation"].copy().sort_values("macro_f1", ascending=False)
    val_class = class_df[class_df["split"] == "validation"].copy()
    train_overall = overall_df[overall_df["split"] == "train_loso"].copy().sort_values("macro_f1", ascending=False)

    header = "Final Pipeline (3-class) — " + ("Grid-Optimized per Modality" if args.optimize else "Defaults")
    report = [
        f"# Modality Tables — {header}",
        f"- feature parquet: `{feature_fp}`",
        f"- optimization: `{args.optimize}`, selection: train_loso_macro_f1",
        "",
        "## Table 1 — Validation Overall Metrics by Modality",
        "",
        val_overall.to_markdown(index=False),
        "",
        "## Table 2 — Validation Classwise Metrics by Modality",
        "",
        val_class.to_markdown(index=False),
        "",
        "## Table 3 — Train-LOSO Overall Metrics by Modality",
        "",
        train_overall.to_markdown(index=False),
    ]
    if best_config_rows:
        best_df = pd.DataFrame(best_config_rows)
        report += ["", "## Table 4 — Selected Config per Modality", "", best_df.to_markdown(index=False)]
    (out_dir / "report.md").write_text("\n".join(report))
    print("\n== Validation ==")
    print(val_overall.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature-parquet", type=Path, default=None)
    p.add_argument("--resample-tag", default="1022")
    p.add_argument("--output-dir", default="results/final/modality_tables")
    p.add_argument("--optimize", action="store_true", help="medium grid search per modality, select by train_loso macro-F1")
    main(p.parse_args())
