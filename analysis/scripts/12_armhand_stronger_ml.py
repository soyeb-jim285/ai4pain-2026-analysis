"""Stronger ML for arm-vs-hand localisation in the AI4Pain 2026 dataset.

Three model families evaluated with LOSO CV on train (41 subjects) and final
held-out validation (12 subjects):

    1. Feature-based classifiers (LR / RF / XGB) on the fully merged
       feature table (physio + tf + raw_stats + temporal + morphology +
       reactivity + multiscale). Subject-z normalisation applied.
    2. Sub-window ensemble: split each 10-s segment into 2-s windows with
       50% overlap, compute per-window stats (mean/std/RMS/skew/kurt/zcr/p2p),
       concatenate, then LR.
    3. 1-D CNN on the raw (4, 1000) tensor with per-subject z-score +
       dropout. Requires torch; `--skip-cnn` disables it.

Only PainArm + PainHand segments are used.

Outputs
-------
- results/tables/armhand_ml_results.csv
- results/tables/armhand_ml_loso_perfold.csv
- results/tables/armhand_ml_validation_eval.csv
- results/tables/armhand_ml_feature_importance_top30.csv
- plots/armhand_ml/loso_macro_f1_by_model.png
- plots/armhand_ml/confusion_matrix_{model}.png
- plots/armhand_ml/per_subject_accuracy_{model}.png
- results/reports/12_armhand_stronger_ml_summary.md

Usage
-----
    uv run python scripts/12_armhand_stronger_ml.py
    uv run python scripts/12_armhand_stronger_ml.py --skip-cnn
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

# Force a non-interactive matplotlib backend before any pyplot import.
# Otherwise RF / XGB multiprocessing workers inherit a Tk connection
# and crash with "Tcl_AsyncDelete: async handler deleted by the wrong thread".
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.data_loader import load_split  # noqa: E402

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
PLOT_DIR = ANALYSIS / "plots" / "armhand_ml"
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
ARM_HAND_CLASSES = ("PainArm", "PainHand")
LABEL_MAP = {"PainArm": 0, "PainHand": 1}


# ---------------------------------------------------------------------------
# Feature loading / merging
# ---------------------------------------------------------------------------
def _read_if_exists(fp: Path) -> pd.DataFrame | None:
    return pd.read_parquet(fp) if fp.exists() else None


def load_enriched_features() -> pd.DataFrame:
    """Combine every feature parquet we have into one wide table."""
    candidates = [
        "all_features_merged.parquet",
        "temporal_features.parquet",
        "morphology_coupling_features.parquet",
        "reactivity_features.parquet",
        "reactivity_interaction_features.parquet",
        "multiscale_features.parquet",
        "multiscale_reactivity_features.parquet",
    ]
    frames = []
    for name in candidates:
        df = _read_if_exists(TAB_DIR / name)
        if df is None:
            continue
        feat_cols = [c for c in df.columns if c not in META_COLS]
        if not feat_cols:
            continue
        meta_present = [c for c in META_COLS if c in df.columns]
        df = df[meta_present + feat_cols].rename(
            columns={c: f"{c}__{Path(name).stem}" for c in feat_cols}
        )
        frames.append(df)

    if not frames:
        raise RuntimeError("no feature parquets found in results/tables/")

    base = frames[0]
    for f in frames[1:]:
        drop_extra = [c for c in META_COLS if c in f.columns and c != "segment_id"]
        f_merge = f.drop(columns=drop_extra, errors="ignore")
        base = base.merge(f_merge, on="segment_id", how="left")
    return base


def prepare_armhand_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep PainArm + PainHand only, drop high-NaN / zero-var, median-impute."""
    df = df[df["class"].isin(ARM_HAND_CLASSES)].copy().reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    nan_frac = df[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if nan_frac[c] <= 0.10]
    X = df[feat_cols].astype(np.float64)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=feat_cols)
    var = X.var(axis=0)
    keep = var.index[var > 1e-12].tolist()
    X = X[keep]
    out = pd.concat([df[META_COLS].reset_index(drop=True),
                     X.astype(np.float32).reset_index(drop=True)], axis=1)
    return out, keep


def apply_subject_z(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    means = out.groupby("subject")[feat_cols].transform("mean")
    stds = out.groupby("subject")[feat_cols].transform("std", ddof=0)
    stds = stds.where(stds > 0, 1.0)
    out[feat_cols] = (out[feat_cols] - means) / stds
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Feature-based LOSO
# ---------------------------------------------------------------------------
def _make_feat_model(name: str):
    if name == "logreg":
        return LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            max_iter=3000, solver="lbfgs", n_jobs=1, random_state=SEED,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=SEED,
        )
    if name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed")
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            max_bin=128, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=4, verbosity=0,
        )
    raise ValueError(name)


def run_loso_features(
    df: pd.DataFrame, feat_cols: list[str], model_name: str
) -> tuple[dict, pd.DataFrame, np.ndarray, pd.Series]:
    train = df[df["split"] == "train"].reset_index(drop=True)
    subjects = train["subject"].to_numpy()
    y = train["class"].map(LABEL_MAP).to_numpy()
    X = train[feat_cols].to_numpy()
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups=subjects))

    per_fold = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    importances = np.zeros(len(feat_cols))
    imp_seen = 0

    for i, (tr, te) in enumerate(
        tqdm(folds, desc=f"LOSO {model_name}", leave=False)
    ):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        mdl = _make_feat_model(model_name)
        mdl.fit(X_tr, y[tr])
        yhat = mdl.predict(X_te)
        per_fold.append({
            "model": model_name, "fold": i,
            "held_out_subject": int(subjects[te][0]),
            "n_test": int(len(te)),
            "acc": accuracy_score(y[te], yhat),
            "macro_f1": f1_score(y[te], yhat, average="macro", zero_division=0),
            "balanced_acc": balanced_accuracy_score(y[te], yhat),
        })
        all_true.append(y[te])
        all_pred.append(yhat)
        try:
            if model_name == "rf":
                importances += mdl.feature_importances_
            elif model_name == "xgb":
                score = mdl.get_booster().get_score(importance_type="gain")
                v = np.zeros(len(feat_cols))
                for k, s in score.items():
                    idx = int(k[1:]) if k.startswith("f") else feat_cols.index(k)
                    v[idx] = s
                importances += v / (v.sum() + 1e-12)
            elif model_name == "logreg":
                importances += np.abs(mdl.coef_).ravel()
            imp_seen += 1
        except Exception:
            pass

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    summary = {
        "model": model_name,
        "loso_acc_mean": float(np.mean([r["acc"] for r in per_fold])),
        "loso_acc_std": float(np.std([r["acc"] for r in per_fold])),
        "loso_macro_f1_mean": float(np.mean([r["macro_f1"] for r in per_fold])),
        "loso_macro_f1_std": float(np.std([r["macro_f1"] for r in per_fold])),
        "loso_balanced_acc_mean": float(
            np.mean([r["balanced_acc"] for r in per_fold])
        ),
        "n_folds": len(per_fold),
        "n_features": len(feat_cols),
    }
    imp = pd.Series(importances / max(imp_seen, 1), index=feat_cols)
    return summary, pd.DataFrame(per_fold), cm, imp


def validation_score_features(
    df: pd.DataFrame, feat_cols: list[str], model_name: str
) -> dict:
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    if val.empty:
        return {"model": model_name, "available": False}
    y_tr = train["class"].map(LABEL_MAP).to_numpy()
    y_va = val["class"].map(LABEL_MAP).to_numpy()
    X_tr = train[feat_cols].to_numpy()
    X_va = val[feat_cols].to_numpy()
    sc = StandardScaler().fit(X_tr)
    X_tr, X_va = sc.transform(X_tr), sc.transform(X_va)
    mdl = _make_feat_model(model_name)
    mdl.fit(X_tr, y_tr)
    yhat = mdl.predict(X_va)
    return {
        "model": model_name, "available": True,
        "val_acc": float(accuracy_score(y_va, yhat)),
        "val_macro_f1": float(f1_score(y_va, yhat, average="macro",
                                       zero_division=0)),
        "val_balanced_acc": float(balanced_accuracy_score(y_va, yhat)),
        "n_val": int(len(y_va)),
    }


# ---------------------------------------------------------------------------
# Sub-window ensemble features
# ---------------------------------------------------------------------------
@dataclass
class WindowConfig:
    n_samples_per_window: int = 200
    stride: int = 100


def extract_subwindow_features(
    tensor: np.ndarray, cfg: WindowConfig
) -> np.ndarray:
    from scipy import stats as sstats

    N, C, T = tensor.shape
    rows = []
    for i in range(N):
        feats: list[float] = []
        for w_start in range(0, T - cfg.n_samples_per_window + 1, cfg.stride):
            w = tensor[i, :, w_start:w_start + cfg.n_samples_per_window]
            for c in range(C):
                x = w[c]
                valid = ~np.isnan(x)
                x = x[valid] if valid.any() else np.array([np.nan])
                if x.size < 3:
                    feats.extend([np.nan] * 7)
                    continue
                mu = np.mean(x)
                sd = np.std(x)
                feats.extend([
                    float(mu),
                    float(sd),
                    float(np.sqrt(np.mean(x * x))),
                    float(np.max(x) - np.min(x)),
                    float(sstats.skew(x)) if sd > 0 else 0.0,
                    float(sstats.kurtosis(x)) if sd > 0 else 0.0,
                    float(np.mean(np.diff(np.signbit(x - mu).astype(int)) != 0))
                    if x.size > 1 else 0.0,
                ])
        rows.append(feats)
    arr = np.array(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def run_subwindow_ensemble(
    train_tensor: np.ndarray, train_meta: pd.DataFrame,
    val_tensor: np.ndarray | None, val_meta: pd.DataFrame | None,
) -> tuple[dict, pd.DataFrame, np.ndarray, dict]:
    cfg = WindowConfig()

    # Arm+Hand only (train)
    mask_tr = train_meta["class"].isin(ARM_HAND_CLASSES).to_numpy()
    Xtr = extract_subwindow_features(train_tensor[mask_tr], cfg)
    meta_tr = train_meta[mask_tr].reset_index(drop=True)
    print(f"[subwindow] train features shape: {Xtr.shape}")
    y = meta_tr["class"].map(LABEL_MAP).to_numpy()
    subjects = meta_tr["subject"].to_numpy()

    logo = LeaveOneGroupOut()
    folds = list(logo.split(Xtr, y, groups=subjects))
    per_fold = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for i, (tr, te) in enumerate(
        tqdm(folds, desc="subwindow LOSO", leave=False)
    ):
        sc = StandardScaler()
        X_tr = sc.fit_transform(Xtr[tr])
        X_te = sc.transform(Xtr[te])
        mdl = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            max_iter=3000, solver="lbfgs", random_state=SEED, n_jobs=1,
        )
        mdl.fit(X_tr, y[tr])
        yhat = mdl.predict(X_te)
        per_fold.append({
            "model": "subwindow_logreg", "fold": i,
            "held_out_subject": int(subjects[te][0]),
            "n_test": int(len(te)),
            "acc": accuracy_score(y[te], yhat),
            "macro_f1": f1_score(y[te], yhat, average="macro", zero_division=0),
            "balanced_acc": balanced_accuracy_score(y[te], yhat),
        })
        all_true.append(y[te])
        all_pred.append(yhat)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    summary = {
        "model": "subwindow_logreg",
        "loso_acc_mean": float(np.mean([r["acc"] for r in per_fold])),
        "loso_acc_std": float(np.std([r["acc"] for r in per_fold])),
        "loso_macro_f1_mean": float(np.mean([r["macro_f1"] for r in per_fold])),
        "loso_macro_f1_std": float(np.std([r["macro_f1"] for r in per_fold])),
        "loso_balanced_acc_mean": float(
            np.mean([r["balanced_acc"] for r in per_fold])
        ),
        "n_folds": len(per_fold),
        "n_features": int(Xtr.shape[1]),
    }

    val_metrics = {"model": "subwindow_logreg", "available": False}
    if val_tensor is not None and val_meta is not None and len(val_meta):
        vmask = val_meta["class"].isin(ARM_HAND_CLASSES).to_numpy()
        Xva = extract_subwindow_features(val_tensor[vmask], cfg)
        meta_va = val_meta[vmask].reset_index(drop=True)
        y_va = meta_va["class"].map(LABEL_MAP).to_numpy()
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        Xva_s = sc.transform(Xva)
        mdl = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            max_iter=3000, solver="lbfgs", random_state=SEED, n_jobs=1,
        )
        mdl.fit(Xtr_s, y)
        yhat = mdl.predict(Xva_s)
        val_metrics = {
            "model": "subwindow_logreg", "available": True,
            "val_acc": float(accuracy_score(y_va, yhat)),
            "val_macro_f1": float(f1_score(y_va, yhat, average="macro",
                                           zero_division=0)),
            "val_balanced_acc": float(balanced_accuracy_score(y_va, yhat)),
            "n_val": int(len(y_va)),
        }
    return summary, pd.DataFrame(per_fold), cm, val_metrics


# ---------------------------------------------------------------------------
# 1-D CNN
# ---------------------------------------------------------------------------
def _build_cnn():
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None, None

    class SmallCNN(nn.Module):
        def __init__(self, in_ch: int = 4, n_classes: int = 2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 32, 7, padding=3),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 3, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    return torch, SmallCNN


def subject_normalise_tensor(
    tensor: np.ndarray, meta: pd.DataFrame
) -> np.ndarray:
    """Per-subject channel z-score using all 36 of the subject's segments."""
    out = tensor.copy()
    for sid in meta["subject"].unique():
        rows = meta.index[meta["subject"] == sid]
        chunk = out[rows]
        for c in range(chunk.shape[1]):
            x = chunk[:, c, :]
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            out[rows, c, :] = ((chunk[:, c, :] - mu) / sd).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0)
    return out


def run_cnn_loso(
    tensor: np.ndarray, meta: pd.DataFrame,
    val_tensor: np.ndarray | None, val_meta: pd.DataFrame | None,
    epochs: int = 25, batch_size: int = 64, lr: float = 1e-3,
) -> tuple[dict, pd.DataFrame, np.ndarray, dict]:
    torch_mod, SmallCNN = _build_cnn()
    if torch_mod is None:
        print("[cnn] torch not available — skipping")
        return ({"model": "cnn1d", "available": False},
                pd.DataFrame(), np.zeros((2, 2), dtype=int),
                {"model": "cnn1d", "available": False})

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cnn] device={device}")

    full_norm = subject_normalise_tensor(tensor, meta)
    mask = meta["class"].isin(ARM_HAND_CLASSES).to_numpy()
    Xn = full_norm[mask]
    meta_ah = meta[mask].reset_index(drop=True)
    y = meta_ah["class"].map(LABEL_MAP).to_numpy().astype(np.int64)
    subjects = meta_ah["subject"].to_numpy()

    logo = LeaveOneGroupOut()
    folds = list(logo.split(Xn, y, groups=subjects))

    per_fold = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for i, (tr, te) in enumerate(
        tqdm(folds, desc="CNN LOSO", leave=False)
    ):
        Xt = torch.from_numpy(Xn[tr]).float()
        Xe = torch.from_numpy(Xn[te]).float()
        yt = torch.from_numpy(y[tr]).long()
        ye = torch.from_numpy(y[te]).long()
        dl = DataLoader(TensorDataset(Xt, yt),
                        batch_size=batch_size, shuffle=True,
                        drop_last=False, num_workers=0)

        model = SmallCNN(in_ch=4, n_classes=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss()
        for ep in range(epochs):
            model.train(True)
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = lossf(logits, yb)
                loss.backward()
                opt.step()

        model.train(False)
        with torch.no_grad():
            logits_te = model(Xe.to(device))
            yhat = logits_te.argmax(dim=1).cpu().numpy()
        per_fold.append({
            "model": "cnn1d", "fold": i,
            "held_out_subject": int(subjects[te][0]),
            "n_test": int(len(te)),
            "acc": accuracy_score(ye.numpy(), yhat),
            "macro_f1": f1_score(ye.numpy(), yhat, average="macro",
                                 zero_division=0),
            "balanced_acc": balanced_accuracy_score(ye.numpy(), yhat),
        })
        all_true.append(ye.numpy())
        all_pred.append(yhat)

        # Persist per-fold progress so a later crash doesn't lose completed folds.
        pd.DataFrame(per_fold).to_csv(
            TAB_DIR / "armhand_ml_cnn_perfold_partial.csv", index=False
        )

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    summary = {
        "model": "cnn1d",
        "loso_acc_mean": float(np.mean([r["acc"] for r in per_fold])),
        "loso_acc_std": float(np.std([r["acc"] for r in per_fold])),
        "loso_macro_f1_mean": float(np.mean([r["macro_f1"] for r in per_fold])),
        "loso_macro_f1_std": float(np.std([r["macro_f1"] for r in per_fold])),
        "loso_balanced_acc_mean": float(
            np.mean([r["balanced_acc"] for r in per_fold])
        ),
        "n_folds": len(per_fold),
        "n_features": int(tensor.shape[1] * tensor.shape[2]),
    }

    val_metrics = {"model": "cnn1d", "available": False}
    if val_tensor is not None and val_meta is not None and len(val_meta):
        val_norm = subject_normalise_tensor(val_tensor, val_meta)
        vmask = val_meta["class"].isin(ARM_HAND_CLASSES).to_numpy()
        Xva = val_norm[vmask]
        y_va = val_meta[vmask]["class"].map(LABEL_MAP).to_numpy().astype(np.int64)
        model = SmallCNN(in_ch=4, n_classes=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss()
        Xt = torch.from_numpy(Xn).float()
        yt = torch.from_numpy(y).long()
        dl = DataLoader(TensorDataset(Xt, yt),
                        batch_size=batch_size, shuffle=True,
                        drop_last=False, num_workers=0)
        for ep in range(epochs):
            model.train(True)
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
        model.train(False)
        with torch.no_grad():
            yhat = model(torch.from_numpy(Xva).float().to(device)).argmax(1).cpu().numpy()
        val_metrics = {
            "model": "cnn1d", "available": True,
            "val_acc": float(accuracy_score(y_va, yhat)),
            "val_macro_f1": float(f1_score(y_va, yhat, average="macro",
                                           zero_division=0)),
            "val_balanced_acc": float(balanced_accuracy_score(y_va, yhat)),
            "n_val": int(len(y_va)),
        }
    return summary, pd.DataFrame(per_fold), cm, val_metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_macro_f1(results: list[dict]) -> None:
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["model"], df["loso_macro_f1_mean"],
           yerr=df.get("loso_macro_f1_std", 0), color="#6b8cae",
           alpha=0.85, edgecolor="black")
    ax.axhline(0.5, color="#d62728", linestyle="--", label="chance")
    ax.set_ylabel("LOSO macro-F1 (mean ± std)")
    ax.set_ylim(0, 1)
    ax.set_title("Arm vs Hand — LOSO macro-F1 by model")
    ax.legend()
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "loso_macro_f1_by_model.png", dpi=130)
    plt.close(fig)


def plot_confmat(cm: np.ndarray, model: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=list(ARM_HAND_CLASSES),
                yticklabels=list(ARM_HAND_CLASSES), ax=ax, cbar=True)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{model} — LOSO confusion (row-norm)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"confusion_matrix_{model}.png", dpi=130)
    plt.close(fig)


def plot_per_subject_accuracy(per_fold_df: pd.DataFrame, model: str) -> None:
    import matplotlib.pyplot as plt

    if per_fold_df.empty:
        return
    sub = per_fold_df[per_fold_df["model"] == model].sort_values("acc")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(sub["held_out_subject"].astype(str), sub["acc"], color="#6b8cae")
    ax.axhline(0.5, color="#d62728", linestyle="--", label="chance")
    ax.set_ylabel("LOSO accuracy")
    ax.set_xlabel("held-out subject")
    ax.set_title(f"{model} — per-subject LOSO accuracy")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"per_subject_accuracy_{model}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(skip_cnn: bool = False) -> None:
    t0 = time.time()
    print("[load] train + validation tensors + features ...")
    train_tensor, train_meta = load_split("train")
    val_tensor, val_meta = load_split("validation")

    print("[load] enriched feature tables ...")
    enriched = load_enriched_features()
    df_all, feat_cols = prepare_armhand_features(enriched)
    df_z = apply_subject_z(df_all, feat_cols)
    print(f"[features] {len(feat_cols)} features × {len(df_z)} arm+hand segments")

    results: list[dict] = []
    per_fold_rows: list[pd.DataFrame] = []
    val_rows: list[dict] = []
    importance_rows: list[pd.Series] = []

    def save_progress(note: str) -> None:
        """Persist everything collected so far. Safe to call after every model."""
        pd.DataFrame(results).to_csv(
            TAB_DIR / "armhand_ml_results.csv", index=False
        )
        if per_fold_rows:
            pd.concat(per_fold_rows, ignore_index=True).to_csv(
                TAB_DIR / "armhand_ml_loso_perfold.csv", index=False
            )
        pd.DataFrame(val_rows).to_csv(
            TAB_DIR / "armhand_ml_validation_eval.csv", index=False
        )
        if importance_rows:
            imp_df = pd.concat(importance_rows, axis=1).fillna(0.0)
            blended = (
                imp_df.rank(ascending=True).mean(axis=1).sort_values(ascending=False)
            )
            imp_df["blended_rank_mean"] = blended
            imp_df.sort_values("blended_rank_mean", ascending=False).head(30).to_csv(
                TAB_DIR / "armhand_ml_feature_importance_top30.csv"
            )
        if results:
            plot_macro_f1(results)

        # Lightweight incremental report so the user can always read current state.
        lines = ["# Arm-vs-Hand — stronger ML (script 12)\n"]
        lines.append(f"- Last saved after: **{note}**")
        lines.append(f"- n_features (merged): **{len(feat_cols)}**")
        lines.append(
            f"- n_arm_hand_segments: train={int((df_z['split']=='train').sum())}, "
            f"val={int((df_z['split']=='validation').sum())}"
        )
        lines.append("")
        lines.append("## LOSO macro-F1 (chance = 0.50)\n")
        lines.append("| model | macro-F1 (mean ± std) | balanced acc | n_features |")
        lines.append("|---|---|---|---|")
        for r in results:
            lines.append(
                f"| {r['model']} | "
                f"{r['loso_macro_f1_mean']:.3f} ± {r['loso_macro_f1_std']:.3f} | "
                f"{r['loso_balanced_acc_mean']:.3f} | {r['n_features']} |"
            )
        lines.append("")
        lines.append("## Validation split (chance = 0.50)\n")
        lines.append("| model | val macro-F1 | val acc | val balanced acc |")
        lines.append("|---|---|---|---|")
        for v in val_rows:
            if not v.get("available"):
                continue
            lines.append(
                f"| {v['model']} | {v['val_macro_f1']:.3f} | "
                f"{v['val_acc']:.3f} | {v['val_balanced_acc']:.3f} |"
            )
        lines.append("")
        lines.append("## Notes")
        lines.append("- All feature-based models use per-subject z-scored merged features.")
        lines.append("- The CNN uses per-subject z-scored raw (4, 1000) tensor.")
        lines.append("- Every fold uses strict LOSO on the held-out subject.")
        lines.append(f"- Runtime so far: {time.time() - t0:.1f}s")
        (REPORT_DIR / "12_armhand_stronger_ml_summary.md").write_text("\n".join(lines))
        print(f"[save] progress persisted ({note})")

    for model in ("logreg", "rf", "xgb" if _HAS_XGB else None):
        if model is None:
            continue
        print(f"\n>>> feature model | {model}")
        summ, fold_df, cm, imp = run_loso_features(df_z, feat_cols, model)
        results.append(summ)
        per_fold_rows.append(fold_df)
        val_rows.append(validation_score_features(df_z, feat_cols, model))
        importance_rows.append(imp.rename(model))
        plot_confmat(cm, model)
        plot_per_subject_accuracy(fold_df, model)
        save_progress(f"feature model {model} done")

    print("\n>>> sub-window ensemble | logreg")
    summ, fold_df, cm, val = run_subwindow_ensemble(
        train_tensor, train_meta, val_tensor, val_meta
    )
    results.append(summ)
    per_fold_rows.append(fold_df)
    val_rows.append(val)
    plot_confmat(cm, "subwindow_logreg")
    plot_per_subject_accuracy(fold_df, "subwindow_logreg")
    save_progress("subwindow ensemble done")

    if not skip_cnn:
        print("\n>>> 1-D CNN | per-subject normalised raw tensor")
        summ, fold_df, cm, val = run_cnn_loso(
            train_tensor, train_meta, val_tensor, val_meta
        )
        if summ.get("available") is False:
            print("[cnn] skipped (torch unavailable)")
        else:
            results.append(summ)
            per_fold_rows.append(fold_df)
            val_rows.append(val)
            plot_confmat(cm, "cnn1d")
            plot_per_subject_accuracy(fold_df, "cnn1d")
            save_progress("cnn done")
    else:
        print("[cnn] skipped by --skip-cnn")

    save_progress("all models complete")
    print(f"\n[save] {REPORT_DIR / '12_armhand_stronger_ml_summary.md'}")
    print(f"[save] {TAB_DIR / 'armhand_ml_results.csv'}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cnn", action="store_true",
                    help="Skip the 1-D CNN (useful if torch is not installed).")
    args = ap.parse_args()
    main(skip_cnn=args.skip_cnn)
