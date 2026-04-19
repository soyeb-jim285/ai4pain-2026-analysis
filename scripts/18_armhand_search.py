"""Resumable ML hyperparameter + feature-set + preprocessing sweep
for the AI4Pain 2026 ARM-vs-HAND localisation problem.

Targets: best macro-F1 with classical ML (no deep learning, no CNN).
Building on Tier A findings:
  - RESP carries the signal (17/53 features FDR-significant)
  - Top features ranked by |Cliff's delta| are mostly RESP

Search space:
  feature_set  in {all, top80_cliff, resp_only, resp_top20, top40_anova_f}
  preproc      in {subject_z+std, subject_z+robust, subject_z+pca20}
  model        in {LR L2, LR L1, LR EN, RF, ExtraTrees, HistGB, XGB, SVM linear, SVM RBF, kNN, GaussianNB}
  hyperparams  per model, small grid

Plus a stacked ensemble built from top-N base learners after all configs
finish.

Persistence
-----------
- results/tables/armhand_search_perfold.csv     (append-only, one row per (config, fold))
- results/tables/armhand_search_progress.json   ({config_id: status})
- results/tables/armhand_search_summary.csv     (rewritten after each config)
- results/tables/armhand_search_validation.csv  (per-config validation metrics)
- results/tables/armhand_search_top.csv         (final top-K configs)
- results/reports/18_armhand_search_summary.md
- plots/armhand_search/

Resume
------
Re-running the script reads the perfold CSV, identifies completed
(config_id, fold) pairs, and skips them. Configs with all 41 folds done
also skip the LOSO step and only re-run validation if missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import RobustScaler, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from tqdm import tqdm  # noqa: E402

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

warnings.filterwarnings("ignore")

SEED = 42
ANALYSIS = Path(__file__).resolve().parents[1]
TAB_DIR = ANALYSIS / "results" / "tables"
REPORT_DIR = ANALYSIS / "results" / "reports"
PLOT_DIR = ANALYSIS / "plots" / "armhand_search"
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

PERFOLD_FP = TAB_DIR / "armhand_search_perfold.csv"
SUMMARY_FP = TAB_DIR / "armhand_search_summary.csv"
VALIDATION_FP = TAB_DIR / "armhand_search_validation.csv"
TOP_FP = TAB_DIR / "armhand_search_top.csv"
REPORT_FP = REPORT_DIR / "18_armhand_search_summary.md"

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
ARM_HAND = ("PainArm", "PainHand")
LABEL_MAP = {"PainArm": 0, "PainHand": 1}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------
def load_features() -> tuple[pd.DataFrame, list[str]]:
    fp = TAB_DIR / "all_features_merged.parquet"
    df = pd.read_parquet(fp)
    df = df[df["class"].isin(("NoPain",) + ARM_HAND)].copy().reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    nan_frac = df[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if nan_frac[c] <= 0.10]
    X = df[feat_cols].astype(np.float64)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                     columns=feat_cols)
    var = X.var(axis=0)
    feat_cols = var.index[var > 1e-12].tolist()
    X = X[feat_cols].astype(np.float32)
    df = pd.concat([df[META_COLS].reset_index(drop=True),
                    X.reset_index(drop=True)], axis=1)
    return df, feat_cols


def apply_subject_z(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    means = out.groupby("subject")[feat_cols].transform("mean")
    stds = out.groupby("subject")[feat_cols].transform("std", ddof=0)
    stds = stds.where(stds > 0, 1.0)
    out[feat_cols] = ((out[feat_cols] - means) / stds).fillna(0.0).astype(np.float32)
    return out


def channel_of(name: str) -> str:
    n = name.lower()
    hits = []
    for tag in ("bvp", "eda", "resp", "spo2", "spo_2"):
        if tag in n:
            hits.append("spo2" if tag == "spo_2" else tag)
    hits = sorted(set(hits))
    if not hits:
        return "other"
    if len(hits) == 1:
        return hits[0]
    return "cross"


# ---------------------------------------------------------------------------
# Feature-set strategies
# ---------------------------------------------------------------------------
def feature_strategies(df_pain: pd.DataFrame, feat_cols: list[str]) -> dict[str, list[str]]:
    """Return mapping name -> ordered list of features.

    Computes Cliff's delta and ANOVA F on TRAIN only; uses outputs from
    Tier A #5 if available."""
    train = df_pain[df_pain["split"] == "train"].reset_index(drop=True)
    arm_means = (train[train["class"] == "PainArm"]
                 .groupby("subject")[feat_cols].mean())
    hand_means = (train[train["class"] == "PainHand"]
                  .groupby("subject")[feat_cols].mean())
    common = arm_means.index.intersection(hand_means.index)
    arm_means = arm_means.loc[common]
    hand_means = hand_means.loc[common]

    cliff = {}
    for f in feat_cols:
        a = arm_means[f].to_numpy()
        h = hand_means[f].to_numpy()
        v = ~(np.isnan(a) | np.isnan(h))
        if v.sum() < 5:
            cliff[f] = 0.0
            continue
        a, h = a[v], h[v]
        diffs = a[:, None] - h[None, :]
        cliff[f] = float((np.sum(diffs > 0) - np.sum(diffs < 0))
                         / (a.size * h.size))
    cliff_series = pd.Series(cliff).abs().sort_values(ascending=False)

    # ANOVA F (paired two-sample) on subject-mean values
    F_vals = {}
    for f in feat_cols:
        a = arm_means[f].dropna().to_numpy()
        h = hand_means[f].dropna().to_numpy()
        if a.size < 5 or h.size < 5:
            F_vals[f] = 0.0
            continue
        try:
            F, _ = f_classif(np.concatenate([a, h]).reshape(-1, 1),
                             np.array([0] * len(a) + [1] * len(h)))
            F_vals[f] = float(F[0])
        except Exception:
            F_vals[f] = 0.0
    f_series = pd.Series(F_vals).sort_values(ascending=False)

    resp_features = [c for c in feat_cols if channel_of(c) == "resp"]
    resp_top20 = (cliff_series.loc[cliff_series.index.intersection(resp_features)]
                  .head(20).index.tolist())

    return {
        "all": feat_cols,
        "top40_cliff": cliff_series.head(40).index.tolist(),
        "top80_cliff": cliff_series.head(80).index.tolist(),
        "top160_cliff": cliff_series.head(min(160, len(feat_cols))).index.tolist(),
        "top40_anova": f_series.head(40).index.tolist(),
        "resp_only": resp_features,
        "resp_top20": resp_top20,
    }


# ---------------------------------------------------------------------------
# Preprocessing pipelines
# ---------------------------------------------------------------------------
def make_preproc(name: str):
    if name == "std":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "pca20":
        return ("compose", StandardScaler(), PCA(n_components=20, random_state=SEED))
    raise ValueError(name)


def transform(preproc, X_train: np.ndarray, X_test: np.ndarray):
    if isinstance(preproc, tuple):
        # ("compose", scaler, pca)
        _, sc, pca = preproc
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        n_comp = min(pca.n_components, X_train.shape[1])
        if n_comp != pca.n_components:
            pca = PCA(n_components=n_comp, random_state=SEED)
        else:
            pca = PCA(n_components=pca.n_components, random_state=SEED)
        pca.fit(X_train)
        return pca.transform(X_train), pca.transform(X_test)
    sc = preproc.__class__()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def model_factory(model_name: str, params: dict):
    p = dict(params)  # copy
    if model_name == "lr_l2":
        return LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight="balanced",
            max_iter=4000, n_jobs=1, random_state=SEED, **p,
        )
    if model_name == "lr_l1":
        return LogisticRegression(
            penalty="l1", solver="liblinear", class_weight="balanced",
            max_iter=4000, random_state=SEED, **p,
        )
    if model_name == "lr_en":
        return LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            max_iter=4000, n_jobs=1, random_state=SEED, **p,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_jobs=-1, class_weight="balanced", random_state=SEED, **p,
        )
    if model_name == "et":
        return ExtraTreesClassifier(
            n_jobs=-1, class_weight="balanced", random_state=SEED, **p,
        )
    if model_name == "histgb":
        return HistGradientBoostingClassifier(
            random_state=SEED, class_weight="balanced", **p,
        )
    if model_name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost missing")
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=4, verbosity=0,
            random_state=SEED, **p,
        )
    if model_name == "svm_linear":
        return SVC(kernel="linear", class_weight="balanced", random_state=SEED,
                   probability=False, **p)
    if model_name == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced", random_state=SEED,
                   probability=False, **p)
    if model_name == "knn":
        return KNeighborsClassifier(n_jobs=-1, **p)
    if model_name == "gnb":
        return GaussianNB(**p)
    raise ValueError(model_name)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
def all_configs() -> list[dict]:
    """Focused, time-bounded search.

    Design: instead of a full grid (which would explode XGB and RF), we
    keep small but representative grids per model family and only fan
    out across feature-set / preproc for the cheapest models.
    """
    cfgs: list[dict] = []
    feature_sets = ["all", "top40_cliff", "top80_cliff",
                    "top40_anova", "resp_only", "resp_top20"]
    cheap_preprocs = ["std", "robust"]

    grids: dict[str, list[dict]] = {
        "lr_l2": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0)],
        "lr_l1": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        "lr_en": [{"C": c, "l1_ratio": r}
                  for c in (0.1, 1.0, 10.0)
                  for r in (0.3, 0.5, 0.7)],
        # RF: 4 representative combos (was 12)
        "rf": [
            {"n_estimators": 300, "max_depth": None, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": 12, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": None, "max_features": 0.3},
            {"n_estimators": 600, "max_depth": None, "max_features": "sqrt"},
        ],
        # ExtraTrees: 2 combos (was 4)
        "et": [
            {"n_estimators": 400, "max_depth": None, "max_features": "sqrt"},
            {"n_estimators": 400, "max_depth": 12, "max_features": 0.3},
        ],
        # HistGB: 4 representative combos (was 12)
        "histgb": [
            {"max_iter": 200, "max_depth": 3, "learning_rate": 0.05},
            {"max_iter": 200, "max_depth": 6, "learning_rate": 0.05},
            {"max_iter": 400, "max_depth": 3, "learning_rate": 0.05},
            {"max_iter": 200, "max_depth": None, "learning_rate": 0.05},
        ],
        # XGB: 6 representative combos (was 48)
        "xgb": ([
            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
             "max_bin": 128, "subsample": 1.0, "colsample_bytree": 1.0},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08,
             "max_bin": 128, "subsample": 1.0, "colsample_bytree": 1.0},
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
             "max_bin": 128, "subsample": 0.8, "colsample_bytree": 0.8},
            {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
             "max_bin": 128, "subsample": 1.0, "colsample_bytree": 1.0},
            {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05,
             "max_bin": 128, "subsample": 0.8, "colsample_bytree": 1.0},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
             "max_bin": 128, "subsample": 0.8, "colsample_bytree": 0.8},
        ] if _HAS_XGB else []),
        "svm_linear": [{"C": c} for c in (0.1, 1.0, 10.0)],
        # SVM RBF: 3 combos, NO probability=True (much faster)
        "svm_rbf": [{"C": c, "gamma": g}
                    for c in (1.0, 10.0)
                    for g in ("scale", 0.1)][:3],
        "knn": [{"n_neighbors": k, "weights": w}
                for k in (11, 21, 41)
                for w in ("uniform", "distance")],
        "gnb": [{}],
    }

    main_fs = "top80_cliff"
    main_pp = "std"
    for model_name, grid in grids.items():
        for params in grid:
            cfgs.append({"model": model_name, "params": params,
                         "feature_set": main_fs, "preproc": main_pp})

    # Cross feature_set × preproc fan-out only for the fast/effective families
    family_cores = {
        "lr_l2": [{"C": 1.0}],
        "lr_l1": [{"C": 1.0}],
        "rf": [{"n_estimators": 300, "max_depth": None, "max_features": "sqrt"}],
        "histgb": [{"max_iter": 200, "max_depth": 6, "learning_rate": 0.05}],
        "xgb": ([{"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08,
                  "max_bin": 128, "subsample": 1.0, "colsample_bytree": 1.0}]
                if _HAS_XGB else []),
    }
    for model_name, glist in family_cores.items():
        for params in glist:
            for fs in feature_sets:
                for pp in cheap_preprocs:
                    if fs == main_fs and pp == main_pp:
                        continue
                    cfgs.append({"model": model_name, "params": params,
                                 "feature_set": fs, "preproc": pp})

    # Add config_id
    for c in cfgs:
        key = json.dumps(
            {"model": c["model"], "params": c["params"],
             "feature_set": c["feature_set"], "preproc": c["preproc"]},
            sort_keys=True, default=str,
        )
        c["config_id"] = hashlib.md5(key.encode()).hexdigest()[:12]
        c["params_str"] = json.dumps(c["params"], sort_keys=True, default=str)
    # de-duplicate
    seen = set()
    unique = []
    for c in cfgs:
        if c["config_id"] in seen:
            continue
        seen.add(c["config_id"])
        unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
PERFOLD_HEADER = [
    "config_id", "model", "params_str", "feature_set", "preproc",
    "fold", "held_out_subject", "n_test", "acc", "macro_f1", "balanced_acc",
    "elapsed_s",
]


def load_completed_folds() -> dict[str, set[int]]:
    if not PERFOLD_FP.exists():
        return {}
    try:
        df = pd.read_csv(PERFOLD_FP)
    except Exception:
        return {}
    out: dict[str, set[int]] = {}
    for cid, sub in df.groupby("config_id"):
        out[cid] = set(int(x) for x in sub["fold"].unique())
    return out


def append_perfold(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)[PERFOLD_HEADER]
    write_header = not PERFOLD_FP.exists()
    df.to_csv(PERFOLD_FP, mode="a", header=write_header, index=False)


def write_summary(perfold: pd.DataFrame, val_df: pd.DataFrame | None) -> pd.DataFrame:
    if perfold.empty:
        return pd.DataFrame()
    g = perfold.groupby(["config_id", "model", "params_str",
                         "feature_set", "preproc"])
    summ = g["macro_f1"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "loso_macro_f1_mean",
                 "std": "loso_macro_f1_std",
                 "count": "n_folds"}
    )
    summ["loso_acc_mean"] = g["acc"].mean()
    summ["loso_balanced_acc_mean"] = g["balanced_acc"].mean()
    summ["elapsed_s_total"] = g["elapsed_s"].sum()
    summ = summ.reset_index()
    if val_df is not None and not val_df.empty:
        summ = summ.merge(val_df, on="config_id", how="left")
    summ = summ.sort_values("loso_macro_f1_mean", ascending=False)
    summ.to_csv(SUMMARY_FP, index=False)
    return summ


def load_validation_results() -> pd.DataFrame:
    if not VALIDATION_FP.exists():
        return pd.DataFrame(
            columns=["config_id", "val_macro_f1", "val_acc", "val_balanced_acc"]
        )
    return pd.read_csv(VALIDATION_FP)


def append_validation(row: dict) -> None:
    df = load_validation_results()
    df = df[df["config_id"] != row["config_id"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(VALIDATION_FP, index=False)


# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------
def run_config(
    cfg: dict, df_z: pd.DataFrame, fs_map: dict[str, list[str]],
    completed_folds: set[int],
) -> tuple[list[dict], dict | None]:
    feat_cols = fs_map[cfg["feature_set"]]
    train = df_z[df_z["split"] == "train"].reset_index(drop=True)
    val = df_z[df_z["split"] == "validation"].reset_index(drop=True)
    train_ah = train[train["class"].isin(ARM_HAND)].reset_index(drop=True)
    val_ah = val[val["class"].isin(ARM_HAND)].reset_index(drop=True)

    X = train_ah[feat_cols].to_numpy()
    y = train_ah["class"].map(LABEL_MAP).to_numpy()
    subjects = train_ah["subject"].to_numpy()

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups=subjects))

    new_rows: list[dict] = []
    for i, (tr, te) in enumerate(folds):
        if i in completed_folds:
            continue
        t0 = time.time()
        try:
            preproc = make_preproc(cfg["preproc"])
            X_tr, X_te = transform(preproc, X[tr], X[te])
            mdl = model_factory(cfg["model"], cfg["params"])
            mdl.fit(X_tr, y[tr])
            yhat = mdl.predict(X_te)
            new_rows.append({
                "config_id": cfg["config_id"],
                "model": cfg["model"], "params_str": cfg["params_str"],
                "feature_set": cfg["feature_set"], "preproc": cfg["preproc"],
                "fold": i,
                "held_out_subject": int(subjects[te][0]),
                "n_test": int(len(te)),
                "acc": accuracy_score(y[te], yhat),
                "macro_f1": f1_score(y[te], yhat, average="macro",
                                     zero_division=0),
                "balanced_acc": balanced_accuracy_score(y[te], yhat),
                "elapsed_s": time.time() - t0,
            })
        except Exception as e:
            new_rows.append({
                "config_id": cfg["config_id"],
                "model": cfg["model"], "params_str": cfg["params_str"],
                "feature_set": cfg["feature_set"], "preproc": cfg["preproc"],
                "fold": i,
                "held_out_subject": int(subjects[te][0]),
                "n_test": int(len(te)),
                "acc": float("nan"),
                "macro_f1": float("nan"),
                "balanced_acc": float("nan"),
                "elapsed_s": time.time() - t0,
            })
        # Stream-flush every fold so a kill -9 loses at most this fold.
        append_perfold([new_rows[-1]])

    # Validation eval (only if not already saved)
    val_row = None
    val_existing = load_validation_results()
    has_val = (cfg["config_id"] in val_existing["config_id"].astype(str).values)
    if not has_val and not val_ah.empty:
        try:
            X_va = val_ah[feat_cols].to_numpy()
            y_va = val_ah["class"].map(LABEL_MAP).to_numpy()
            preproc = make_preproc(cfg["preproc"])
            X_tr_full, X_va_t = transform(preproc, X, X_va)
            mdl = model_factory(cfg["model"], cfg["params"])
            mdl.fit(X_tr_full, y)
            yhat = mdl.predict(X_va_t)
            val_row = {
                "config_id": cfg["config_id"],
                "val_macro_f1": float(f1_score(y_va, yhat, average="macro",
                                               zero_division=0)),
                "val_acc": float(accuracy_score(y_va, yhat)),
                "val_balanced_acc": float(balanced_accuracy_score(y_va, yhat)),
                "n_val": int(len(y_va)),
            }
            append_validation(val_row)
        except Exception:
            val_row = None
    return new_rows, val_row


# ---------------------------------------------------------------------------
# Ensemble of top-K
# ---------------------------------------------------------------------------
def ensemble_top_k(
    summary: pd.DataFrame, df_z: pd.DataFrame, fs_map: dict[str, list[str]],
    k: int = 5,
) -> dict:
    """Average predicted probabilities of the top-K configs by LOSO macro-F1.
    Re-runs LOSO so we can collect per-fold probabilities; only top-K configs
    so this is fast.
    """
    top = summary.head(k).copy()
    print(f"[ensemble] using top-{k} configs:")
    for _, r in top.iterrows():
        print(f"  - {r['model']:<10} fs={r['feature_set']:<14} pp={r['preproc']:<6} "
              f"loso={r['loso_macro_f1_mean']:.3f}")

    train = df_z[df_z["split"] == "train"].reset_index(drop=True)
    val = df_z[df_z["split"] == "validation"].reset_index(drop=True)
    train_ah = train[train["class"].isin(ARM_HAND)].reset_index(drop=True)
    val_ah = val[val["class"].isin(ARM_HAND)].reset_index(drop=True)

    subjects = train_ah["subject"].to_numpy()
    y_tr = train_ah["class"].map(LABEL_MAP).to_numpy()
    y_va = val_ah["class"].map(LABEL_MAP).to_numpy() if not val_ah.empty else None

    logo = LeaveOneGroupOut()
    n = len(train_ah)
    fold_idx = list(logo.split(np.zeros(n), y_tr, groups=subjects))

    proba_loso_per_cfg = []
    proba_val_per_cfg = []
    for _, r in top.iterrows():
        cfg = {
            "model": r["model"], "params": json.loads(r["params_str"]),
            "feature_set": r["feature_set"], "preproc": r["preproc"],
        }
        feat_cols = fs_map[cfg["feature_set"]]
        X = train_ah[feat_cols].to_numpy()
        X_va = val_ah[feat_cols].to_numpy() if not val_ah.empty else None
        loso_proba = np.zeros(n, dtype=np.float64)
        for tr, te in fold_idx:
            preproc = make_preproc(cfg["preproc"])
            X_tr_t, X_te_t = transform(preproc, X[tr], X[te])
            mdl = model_factory(cfg["model"], cfg["params"])
            mdl.fit(X_tr_t, y_tr[tr])
            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(X_te_t)[:, 1]
            else:
                p = (mdl.decision_function(X_te_t) > 0).astype(float)
            loso_proba[te] = p
        proba_loso_per_cfg.append(loso_proba)
        if X_va is not None:
            preproc = make_preproc(cfg["preproc"])
            X_tr_full, X_va_t = transform(preproc, X, X_va)
            mdl = model_factory(cfg["model"], cfg["params"])
            mdl.fit(X_tr_full, y_tr)
            if hasattr(mdl, "predict_proba"):
                pv = mdl.predict_proba(X_va_t)[:, 1]
            else:
                pv = (mdl.decision_function(X_va_t) > 0).astype(float)
            proba_val_per_cfg.append(pv)

    loso_proba_mean = np.mean(proba_loso_per_cfg, axis=0)
    loso_pred = (loso_proba_mean >= 0.5).astype(int)
    out = {
        "k": k,
        "loso_macro_f1": float(f1_score(y_tr, loso_pred, average="macro",
                                        zero_division=0)),
        "loso_acc": float(accuracy_score(y_tr, loso_pred)),
        "loso_balanced_acc": float(balanced_accuracy_score(y_tr, loso_pred)),
    }
    if proba_val_per_cfg:
        val_proba_mean = np.mean(proba_val_per_cfg, axis=0)
        val_pred = (val_proba_mean >= 0.5).astype(int)
        out.update({
            "val_macro_f1": float(f1_score(y_va, val_pred, average="macro",
                                           zero_division=0)),
            "val_acc": float(accuracy_score(y_va, val_pred)),
            "val_balanced_acc": float(balanced_accuracy_score(y_va, val_pred)),
        })
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if summary.empty:
        return
    top20 = summary.head(20).copy()
    top20["label"] = (top20["model"] + " | " + top20["feature_set"]
                      + " | " + top20["preproc"])

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top20["label"][::-1], top20["loso_macro_f1_mean"][::-1],
            xerr=top20["loso_macro_f1_std"][::-1], color="#3b82f6",
            edgecolor="black")
    ax.axvline(0.5, color="r", linestyle="--", label="chance")
    ax.set_xlabel("LOSO macro-F1")
    ax.set_title("Top-20 ARM-vs-HAND configurations")
    ax.legend()
    ax.set_xlim(0.4, 0.7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "top20_loso_macro_f1.png", dpi=130)
    plt.close(fig)

    if "val_macro_f1" in summary.columns:
        sub = summary.dropna(subset=["val_macro_f1"]).copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(sub["loso_macro_f1_mean"], sub["val_macro_f1"], alpha=0.5)
            ax.plot([0.4, 0.7], [0.4, 0.7], "k--", alpha=0.5)
            ax.axhline(0.5, color="r", linestyle=":", alpha=0.5)
            ax.axvline(0.5, color="r", linestyle=":", alpha=0.5)
            ax.set_xlabel("LOSO macro-F1")
            ax.set_ylabel("Validation macro-F1")
            ax.set_title("LOSO vs Validation macro-F1, all configs")
            fig.tight_layout()
            fig.savefig(PLOT_DIR / "loso_vs_val_scatter.png", dpi=130)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if "val_macro_f1" in summary.columns:
        df_box = summary[["model", "loso_macro_f1_mean", "val_macro_f1"]].melt(
            id_vars=["model"], var_name="metric", value_name="macro_f1"
        )
    else:
        df_box = summary[["model", "loso_macro_f1_mean"]].rename(
            columns={"loso_macro_f1_mean": "macro_f1"}).assign(metric="loso")
    sns.boxplot(data=df_box, x="model", y="macro_f1", hue="metric", ax=ax)
    ax.axhline(0.5, color="r", linestyle="--", label="chance")
    ax.set_title("Distribution of macro-F1 per model family")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "macro_f1_per_model_box.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(skip_ensemble: bool = False, max_configs: int | None = None) -> None:
    t0 = time.time()
    print("[load] features ...")
    df_all, feat_cols = load_features()
    print(f"[load] {len(feat_cols)} features × {len(df_all)} segments")

    print("[normalise] subject-z ...")
    df_z = apply_subject_z(df_all, feat_cols)

    print("[fs] computing feature-set rankings ...")
    fs_map = feature_strategies(df_z[df_z["class"].isin(ARM_HAND)], feat_cols)
    for k, v in fs_map.items():
        print(f"  fs={k:<14} n={len(v)}")

    cfgs = all_configs()
    if max_configs is not None:
        cfgs = cfgs[:max_configs]
    print(f"\n[search] {len(cfgs)} configurations to evaluate")

    completed = load_completed_folds()
    val_existing = load_validation_results()
    val_done_ids = set(val_existing["config_id"].astype(str).values)

    todo = []
    skipped = 0
    for cfg in cfgs:
        cf = completed.get(cfg["config_id"], set())
        if len(cf) >= 41 and cfg["config_id"] in val_done_ids:
            skipped += 1
        else:
            todo.append(cfg)
    print(f"[resume] {skipped} configs already complete, {len(todo)} to run\n")

    pbar = tqdm(todo, desc="configs")
    for cfg in pbar:
        pbar.set_postfix({"model": cfg["model"], "fs": cfg["feature_set"],
                          "pp": cfg["preproc"]})
        cf = completed.get(cfg["config_id"], set())
        run_config(cfg, df_z, fs_map, cf)

        # Re-aggregate after each config so the user can read up-to-date summary.
        if PERFOLD_FP.exists():
            try:
                pf = pd.read_csv(PERFOLD_FP)
                val_df = load_validation_results()
                summary = write_summary(pf, val_df)
                if not summary.empty:
                    print(
                        f"  best so far: {summary.iloc[0]['model']:<10} "
                        f"fs={summary.iloc[0]['feature_set']:<14} "
                        f"pp={summary.iloc[0]['preproc']:<6} "
                        f"loso={summary.iloc[0]['loso_macro_f1_mean']:.3f} "
                        f"val={summary.iloc[0].get('val_macro_f1', float('nan')):.3f}"
                    )
            except Exception as e:
                print(f"  (summary write skipped: {e})")

    # Final aggregation
    if PERFOLD_FP.exists():
        pf = pd.read_csv(PERFOLD_FP)
        val_df = load_validation_results()
        summary = write_summary(pf, val_df)
    else:
        summary = pd.DataFrame()

    # Ensemble of top configs
    ens_results: list[dict] = []
    if not skip_ensemble and not summary.empty:
        for k in (3, 5, 10):
            try:
                r = ensemble_top_k(summary, df_z, fs_map, k=k)
                ens_results.append(r)
            except Exception as e:
                print(f"[ensemble k={k}] failed: {e}")

    if ens_results:
        pd.DataFrame(ens_results).to_csv(
            TAB_DIR / "armhand_search_ensembles.csv", index=False
        )

    if not summary.empty:
        summary.head(20).to_csv(TOP_FP, index=False)

    # Plots
    if not summary.empty:
        make_plots(summary)

    # Report
    lines = ["# 18 — ARM vs HAND ML hyperparameter search\n"]
    lines.append(f"- runtime so far: {time.time() - t0:.1f}s")
    lines.append(f"- configurations attempted: {len(cfgs)}")
    if not summary.empty:
        lines.append(f"- configurations with full LOSO + val: "
                     f"{int((summary['n_folds'] == 41).sum())}")
        lines.append("")
        lines.append("## Top-15 configurations (by LOSO macro-F1)\n")
        cols = ["model", "feature_set", "preproc", "params_str",
                "loso_macro_f1_mean", "loso_macro_f1_std",
                "val_macro_f1", "val_acc"]
        cols = [c for c in cols if c in summary.columns]
        lines.append(summary.head(15)[cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    if ens_results:
        lines.append("## Ensembles of top-K configs (mean predicted probability)\n")
        lines.append(pd.DataFrame(ens_results).to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Notes")
    lines.append("- Resumable: rerun the script to continue from the last fold.")
    lines.append("- Per-fold trail: `results/tables/armhand_search_perfold.csv`.")
    lines.append("- Top-K table: `results/tables/armhand_search_top.csv`.")
    REPORT_FP.write_text("\n".join(lines))
    print(f"\n[save] {REPORT_FP}")
    print(f"[save] {SUMMARY_FP}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ensemble", action="store_true")
    ap.add_argument("--max-configs", type=int, default=None,
                    help="Limit number of configs (for smoke testing)")
    args = ap.parse_args()
    main(skip_ensemble=args.skip_ensemble, max_configs=args.max_configs)
