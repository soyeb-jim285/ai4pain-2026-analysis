"""Baseline classifiers for the AI4Pain 2026 dataset.

Leave-One-Subject-Out cross-validation on the train split (41 subjects) with
three label schemes:
    * 3-class  : NoPain vs PainArm vs PainHand
    * binary   : NoPain vs Pain (arm + hand merged)
    * armhand  : PainArm vs PainHand on pain-only segments

Two feature-preprocessing variants:
    * A "global"    : train-fit StandardScaler
    * B "subjectz"  : per-subject z (each subject's mean/std over its 36
                      segments) then train-fit StandardScaler

Three models: LogReg, RandomForest, XGBoost.

Outputs are written under results/tables, plots/baselines and
results/reports.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
ANALYSIS = Path(__file__).resolve().parents[1]
TAB_DIR = ANALYSIS / "results" / "tables"
REPORT_DIR = ANALYSIS / "results" / "reports"
PLOT_DIR = ANALYSIS / "plots" / "baselines"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER_3 = ["NoPain", "PainArm", "PainHand"]
CLASS_ORDER_BIN = ["NoPain", "Pain"]
CLASS_ORDER_AH = ["PainArm", "PainHand"]

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]


# ---------------------------------------------------------------------------
# Data loading / merging
# ---------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    merged_fp = TAB_DIR / "all_features_merged.parquet"
    if merged_fp.exists():
        print(f"Loading merged features: {merged_fp}")
        return pd.read_parquet(merged_fp)

    print("Merging physio/tf/raw stats feature tables on segment_id")
    physio = pd.read_parquet(TAB_DIR / "physio_features.parquet")
    tf = pd.read_parquet(TAB_DIR / "tf_features.parquet")
    raw = pd.read_parquet(TAB_DIR / "raw_stats_per_segment.parquet")

    merged = physio.copy()
    for other, tag in [(tf, "tf"), (raw, "raw")]:
        drop = [c for c in META_COLS if c != "segment_id"]
        other2 = other.drop(columns=drop)
        dup = [c for c in other2.columns if c != "segment_id" and c in merged.columns]
        if dup:
            other2 = other2.drop(columns=dup)
        merged = merged.merge(other2, on="segment_id", how="left")

    merged.to_parquet(merged_fp, index=False)
    return merged


def prep_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop high-NaN cols, median-impute, drop zero-variance cols."""
    feat_cols = [c for c in df.columns if c not in META_COLS]

    # Drop columns with >10% NaN
    nan_frac = df[feat_cols].isna().mean()
    keep_mask = nan_frac <= 0.10
    dropped_nan = nan_frac[~keep_mask].index.tolist()
    feat_cols = [c for c in feat_cols if keep_mask[c]]
    if dropped_nan:
        print(f"Dropped {len(dropped_nan)} features for NaN>10% "
              f"(e.g. {dropped_nan[:5]})")

    X = df[feat_cols].copy()
    med = df.loc[df["split"] == "train", feat_cols].median(numeric_only=True)
    X = X.fillna(med)
    X = X.fillna(df[feat_cols].median(numeric_only=True))
    X = X.fillna(0.0)

    var = X.loc[df["split"] == "train"].var(numeric_only=True)
    zero_var = var[var <= 1e-12].index.tolist()
    if zero_var:
        print(f"Dropped {len(zero_var)} zero-variance columns "
              f"(e.g. {zero_var[:5]})")
        X = X.drop(columns=zero_var)
        feat_cols = [c for c in feat_cols if c not in zero_var]

    X = X.astype(np.float32)
    out = pd.concat([df[META_COLS].reset_index(drop=True),
                     X.reset_index(drop=True)], axis=1)
    return out, feat_cols


# ---------------------------------------------------------------------------
# Preprocessing variants
# ---------------------------------------------------------------------------
def apply_subject_z(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Per-subject z-score each feature using that subject's mean/std.

    Vectorised: compute per-subject mean and std matrices once, then
    broadcast. Handles zero-variance features by leaving the numerator
    (x - mean) and dividing by 1.0 for those features.
    """
    out = df.copy()
    means = out.groupby("subject")[feat_cols].transform("mean")
    stds = out.groupby("subject")[feat_cols].transform("std", ddof=0)
    stds = stds.where(stds > 0, 1.0)
    out[feat_cols] = (out[feat_cols] - means) / stds
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_model(name: str, n_classes: int):
    if name == "logreg":
        return LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            max_iter=2000, n_jobs=-1, random_state=SEED,
            solver="lbfgs",
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            random_state=SEED, class_weight="balanced",
        )
    if name == "xgb":
        if n_classes > 2:
            return XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.08,
                max_bin=128, random_state=SEED, tree_method="hist",
                objective="multi:softprob", num_class=n_classes,
                n_jobs=4, verbosity=0,
            )
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            max_bin=128, random_state=SEED, tree_method="hist",
            objective="binary:logistic",
            n_jobs=4, verbosity=0, eval_metric="logloss",
        )
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def encode_labels(classes: pd.Series, scheme: str) -> tuple[np.ndarray, list[str]]:
    if scheme == "3class":
        lut = {c: i for i, c in enumerate(CLASS_ORDER_3)}
        y = classes.map(lut).to_numpy()
        return y, CLASS_ORDER_3
    if scheme == "binary":
        y = (classes != "NoPain").astype(int).to_numpy()
        return y, CLASS_ORDER_BIN
    if scheme == "armhand":
        lut = {"PainArm": 0, "PainHand": 1}
        y = classes.map(lut).to_numpy()
        return y, CLASS_ORDER_AH
    raise ValueError(scheme)


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------
def importance_from_model(model, feat_cols: list[str]) -> np.ndarray:
    if isinstance(model, RandomForestClassifier):
        return model.feature_importances_
    if isinstance(model, XGBClassifier):
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        vec = np.zeros(len(feat_cols), dtype=np.float64)
        for k, v in scores.items():
            try:
                idx = int(k[1:])
                if 0 <= idx < len(feat_cols):
                    vec[idx] = v
            except ValueError:
                continue
        return vec
    if isinstance(model, LogisticRegression):
        coef = model.coef_
        if coef.ndim == 1:
            return np.abs(coef)
        return np.abs(coef).mean(axis=0)
    return np.zeros(len(feat_cols))


# ---------------------------------------------------------------------------
# Core LOSO assessment for one (model, preproc, label_scheme)
# ---------------------------------------------------------------------------
def run_loso(
    train_df: pd.DataFrame,
    feat_cols: list[str],
    label_scheme: str,
    model_name: str,
    preproc: str,
) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    if label_scheme == "armhand":
        sub = train_df[train_df["class"] != "NoPain"].reset_index(drop=True)
    else:
        sub = train_df.reset_index(drop=True)

    y, class_names = encode_labels(sub["class"], label_scheme)
    subjects = sub["subject"].to_numpy()

    data = sub.copy()
    if preproc == "subjectz":
        data = apply_subject_z(data, feat_cols)

    X_all = data[feat_cols].to_numpy().astype(np.float32)

    n_classes = len(class_names)
    logo = LeaveOneGroupOut()

    per_fold = []
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    imp_accum = np.zeros(len(feat_cols), dtype=np.float64)
    imp_n = 0

    unique_subjs = np.unique(subjects)
    tag = f"{model_name}|{preproc}|{label_scheme}"

    for train_idx, val_idx in tqdm(
        logo.split(X_all, y, groups=subjects),
        total=len(unique_subjs),
        desc=tag,
        leave=False,
    ):
        X_tr, X_va = X_all[train_idx], X_all[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        subj_out = int(subjects[val_idx][0])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = make_model(model_name, n_classes)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_va_s)

        acc = accuracy_score(y_va, y_pred)
        bal_acc = balanced_accuracy_score(y_va, y_pred)
        f1_macro = f1_score(y_va, y_pred, average="macro", zero_division=0)
        per_class_f1 = f1_score(
            y_va, y_pred, labels=list(range(n_classes)),
            average=None, zero_division=0,
        )
        cm = confusion_matrix(y_va, y_pred, labels=list(range(n_classes)))
        conf += cm

        row = {
            "model": model_name,
            "preprocessing": preproc,
            "label_scheme": label_scheme,
            "subject_held_out": subj_out,
            "n_val": int(len(y_va)),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": f1_macro,
        }
        for cn, f in zip(class_names, per_class_f1):
            row[f"f1_{cn}"] = float(f)
        per_fold.append(row)

        imp = importance_from_model(model, feat_cols)
        if np.isfinite(imp).all() and imp.sum() > 0:
            imp_accum += imp
            imp_n += 1

    if imp_n > 0:
        imp_accum /= imp_n

    pf_df = pd.DataFrame(per_fold)
    summary = {
        "model": model_name,
        "preprocessing": preproc,
        "label_scheme": label_scheme,
        "split_eval_type": "loso_train",
        "mean_acc": pf_df["accuracy"].mean(),
        "std_acc": pf_df["accuracy"].std(),
        "macro_f1_mean": pf_df["macro_f1"].mean(),
        "macro_f1_std": pf_df["macro_f1"].std(),
        "balanced_acc_mean": pf_df["balanced_accuracy"].mean(),
        "n_folds": int(len(pf_df)),
    }
    for cn in class_names:
        summary[f"mean_f1_{cn}"] = pf_df[f"f1_{cn}"].mean()
    summary["notes"] = f"LOSO {len(unique_subjs)} folds, classes={class_names}"
    return summary, per_fold, conf, imp_accum


# ---------------------------------------------------------------------------
# Validation-set assessment: fit on all train, test on val
# ---------------------------------------------------------------------------
def run_validation(
    full_df: pd.DataFrame,
    feat_cols: list[str],
    label_scheme: str,
    model_name: str,
    preproc: str,
) -> tuple[dict, np.ndarray]:
    if label_scheme == "armhand":
        sub = full_df[full_df["class"] != "NoPain"].reset_index(drop=True)
    else:
        sub = full_df.reset_index(drop=True)

    data = sub.copy()
    if preproc == "subjectz":
        data = apply_subject_z(data, feat_cols)

    train_mask = (data["split"] == "train").to_numpy()
    val_mask = (data["split"] == "validation").to_numpy()

    y_all, class_names = encode_labels(data["class"], label_scheme)
    n_classes = len(class_names)

    X = data[feat_cols].to_numpy().astype(np.float32)
    X_tr, y_tr = X[train_mask], y_all[train_mask]
    X_va, y_va = X[val_mask], y_all[val_mask]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    model = make_model(model_name, n_classes)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_va_s)

    acc = accuracy_score(y_va, y_pred)
    bal_acc = balanced_accuracy_score(y_va, y_pred)
    f1_macro = f1_score(y_va, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        y_va, y_pred, labels=list(range(n_classes)),
        average=None, zero_division=0,
    )
    cm = confusion_matrix(y_va, y_pred, labels=list(range(n_classes)))

    row = {
        "model": model_name,
        "preprocessing": preproc,
        "label_scheme": label_scheme,
        "split_eval_type": "validation",
        "accuracy": acc,
        "balanced_acc": bal_acc,
        "macro_f1": f1_macro,
        "n_val": int(len(y_va)),
    }
    for cn, f in zip(class_names, per_class_f1):
        row[f"f1_{cn}"] = float(f)
    return row, cm


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_acc_bar(results_df: pd.DataFrame, label_scheme: str, out_fp: Path,
                 chance: float):
    sub = results_df[
        (results_df["label_scheme"] == label_scheme) &
        (results_df["split_eval_type"] == "loso_train")
    ].copy()
    if sub.empty:
        return
    sub["combo"] = sub["model"] + "|" + sub["preprocessing"]
    sub = sub.sort_values("mean_acc", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sub["combo"], sub["mean_acc"],
           yerr=sub["std_acc"], capsize=4,
           color=sns.color_palette("viridis", len(sub)))
    ax.axhline(chance, linestyle="--", color="red", label=f"chance={chance:.2f}")
    ax.set_ylabel("LOSO mean accuracy (±std)")
    ax.set_title(f"LOSO accuracy — {label_scheme}")
    ax.set_xticklabels(sub["combo"], rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)


def plot_confmat(cm: np.ndarray, class_names: list[str], title: str,
                 out_fp: Path):
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)


def plot_per_subject(per_fold_df: pd.DataFrame, model_name: str,
                     label_scheme: str, out_fp: Path):
    sub = per_fold_df[
        (per_fold_df["model"] == model_name) &
        (per_fold_df["label_scheme"] == label_scheme)
    ]
    best_preproc = (
        sub.groupby("preprocessing")["accuracy"].mean().idxmax()
    )
    sub = sub[sub["preprocessing"] == best_preproc].sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sub["subject_held_out"].astype(str), sub["accuracy"],
           color="steelblue")
    ax.axhline(sub["accuracy"].mean(), linestyle="--", color="red",
               label=f"mean={sub['accuracy'].mean():.3f}")
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-subject LOSO acc — {model_name} / {label_scheme} "
                 f"/ {best_preproc}")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)

    merged = load_features()
    print(f"Merged features: {merged.shape}")
    assert set(["split", "subject", "class", "segment_id"]).issubset(merged.columns)

    merged, feat_cols = prep_feature_matrix(merged)
    print(f"After prep: {len(feat_cols)} features, {len(merged)} rows")

    train_df = merged[merged["split"] == "train"].reset_index(drop=True)
    print(f"Train: {len(train_df)} rows, "
          f"{train_df['subject'].nunique()} subjects")

    MODELS = ["logreg", "rf", "xgb"]
    PREPROCS = ["global", "subjectz"]
    LABEL_SCHEMES = ["3class", "binary", "armhand"]

    summaries: list[dict] = []
    all_perfold: list[dict] = []
    all_cms: dict[str, np.ndarray] = {}
    imp_table: dict[tuple, np.ndarray] = {}

    for label_scheme in LABEL_SCHEMES:
        for preproc in PREPROCS:
            for model_name in MODELS:
                print(f"\n>>> {label_scheme} | {preproc} | {model_name}")
                summary, per_fold, cm, imp = run_loso(
                    train_df, feat_cols, label_scheme, model_name, preproc,
                )
                summaries.append(summary)
                all_perfold.extend(per_fold)
                all_cms[f"{model_name}_{preproc}_{label_scheme}"] = cm
                imp_table[(model_name, preproc, label_scheme)] = imp
                print(f"    mean_acc={summary['mean_acc']:.3f} ± "
                      f"{summary['std_acc']:.3f}  "
                      f"macro_f1={summary['macro_f1_mean']:.3f}")

    results_df = pd.DataFrame(summaries)
    per_fold_df = pd.DataFrame(all_perfold)

    print("\n=== Validation-split assessment ===")
    val_rows = []
    val_cms: dict[str, np.ndarray] = {}
    for label_scheme in LABEL_SCHEMES:
        for preproc in PREPROCS:
            for model_name in MODELS:
                row, cm = run_validation(
                    merged, feat_cols, label_scheme, model_name, preproc,
                )
                print(f"    val {label_scheme}/{preproc}/{model_name}: "
                      f"acc={row['accuracy']:.3f}  "
                      f"macro_f1={row['macro_f1']:.3f}")
                val_rows.append(row)
                val_cms[f"{model_name}_{preproc}_{label_scheme}_val"] = cm
    val_df = pd.DataFrame(val_rows)

    results_df.to_csv(TAB_DIR / "baseline_results.csv", index=False)
    per_fold_df.to_csv(TAB_DIR / "baseline_loso_perfold.csv", index=False)
    val_df.to_csv(TAB_DIR / "baseline_validation_eval.csv", index=False)
    np.savez(TAB_DIR / "baseline_confusion_matrices_loso.npz",
             **{k: v for k, v in all_cms.items()},
             **{k: v for k, v in val_cms.items()})

    print("\n=== Feature importance blend ===")
    imp_df_rows = []
    for (model_name, preproc, label_scheme), vec in imp_table.items():
        if label_scheme != "3class":
            continue
        vec = np.asarray(vec, dtype=float)
        if vec.sum() <= 0:
            continue
        order = np.argsort(-vec)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        for i, fn in enumerate(feat_cols):
            imp_df_rows.append({
                "feature": fn,
                "model": model_name,
                "preprocessing": preproc,
                "importance": float(vec[i]),
                "rank": int(ranks[i]),
            })
    imp_df = pd.DataFrame(imp_df_rows)
    blend = (imp_df.groupby("feature")["rank"].mean()
             .sort_values().reset_index()
             .rename(columns={"rank": "mean_rank"}))
    blend_top = blend.head(50).copy()
    for (model_name, preproc, label_scheme), vec in imp_table.items():
        if label_scheme != "3class":
            continue
        if vec.sum() > 0:
            norm = vec / (vec.sum() or 1.0)
            col = f"{model_name}_{preproc}_norm_imp"
            blend_top[col] = [
                float(norm[feat_cols.index(f)]) if f in feat_cols else np.nan
                for f in blend_top["feature"]
            ]
    blend_top.to_csv(TAB_DIR / "baseline_feature_importance_top50.csv",
                     index=False)

    print("\n=== Plots ===")
    plot_acc_bar(results_df, "3class",
                 PLOT_DIR / "loso_accuracy_bar.png", chance=1 / 3)
    plot_acc_bar(results_df, "binary",
                 PLOT_DIR / "loso_accuracy_bar_binary.png", chance=0.5)
    plot_acc_bar(results_df, "armhand",
                 PLOT_DIR / "loso_accuracy_bar_armhand.png", chance=0.5)

    for label_scheme, class_names in [
        ("3class", CLASS_ORDER_3), ("binary", CLASS_ORDER_BIN),
    ]:
        for model_name in MODELS:
            for preproc in PREPROCS:
                key = f"{model_name}_{preproc}_{label_scheme}"
                cm = all_cms.get(key)
                if cm is None:
                    continue
                plot_confmat(
                    cm, class_names,
                    f"LOSO CM {model_name}/{preproc}/{label_scheme}",
                    PLOT_DIR / f"confusion_matrix_loso_{key}.png",
                )

    for m in MODELS:
        plot_per_subject(per_fold_df, m, "3class",
                         PLOT_DIR / f"per_subject_accuracy_{m}_3class.png")

    top20 = blend.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top20["feature"][::-1], (1 / (1 + top20["mean_rank"][::-1])),
            color=sns.color_palette("magma", len(top20)))
    ax.set_xlabel("1 / (1 + mean rank across models×preprocs)")
    ax.set_title("Top 20 features — blended importance (3-class LOSO)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_importance_top20.png", dpi=140)
    plt.close(fig)

    merge_keys = ["model", "preprocessing", "label_scheme"]
    loso_for_scatter = results_df[merge_keys + ["macro_f1_mean"]].rename(
        columns={"macro_f1_mean": "loso_macro_f1"}
    )
    val_for_scatter = val_df[merge_keys + ["macro_f1"]].rename(
        columns={"macro_f1": "val_macro_f1"}
    )
    sc = loso_for_scatter.merge(val_for_scatter, on=merge_keys)
    fig, ax = plt.subplots(figsize=(6, 6))
    for scheme, grp in sc.groupby("label_scheme"):
        ax.scatter(grp["loso_macro_f1"], grp["val_macro_f1"],
                   label=scheme, s=60)
    lo = min(sc["loso_macro_f1"].min(), sc["val_macro_f1"].min()) - 0.02
    hi = max(sc["loso_macro_f1"].max(), sc["val_macro_f1"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("LOSO macro-F1 (train)")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Generalization: LOSO vs. held-out validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "validation_vs_loso_scatter.png", dpi=140)
    plt.close(fig)

    write_report(results_df, per_fold_df, val_df, blend_top, feat_cols)

    print("\n=== FINAL SUMMARY (LOSO mean_acc / macro_f1) ===")
    print(results_df.sort_values(["label_scheme", "macro_f1_mean"],
                                 ascending=[True, False]).to_string(index=False))

    print("\n=== VALIDATION SPLIT ===")
    print(val_df.sort_values(["label_scheme", "macro_f1"],
                             ascending=[True, False]).to_string(index=False))


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def write_report(results_df, per_fold_df, val_df, blend_top, feat_cols):
    def best(df, scheme, metric="macro_f1_mean"):
        sub = df[(df["label_scheme"] == scheme) &
                 (df["split_eval_type"] == "loso_train")]
        return sub.sort_values(metric, ascending=False).iloc[0]

    best3 = best(results_df, "3class")
    bestb = best(results_df, "binary")
    besta = best(results_df, "armhand")

    ab = (results_df[results_df["split_eval_type"] == "loso_train"]
          .groupby(["label_scheme", "preprocessing"])["macro_f1_mean"]
          .mean().unstack())

    best_pf = per_fold_df[
        (per_fold_df["model"] == best3["model"]) &
        (per_fold_df["preprocessing"] == best3["preprocessing"]) &
        (per_fold_df["label_scheme"] == "3class")
    ].sort_values("accuracy").head(5)

    top15 = blend_top.head(15)[["feature", "mean_rank"]]

    val_row = val_df[
        (val_df["model"] == best3["model"]) &
        (val_df["preprocessing"] == best3["preprocessing"]) &
        (val_df["label_scheme"] == "3class")
    ]
    val_f1 = float(val_row["macro_f1"].iloc[0]) if len(val_row) else float("nan")

    lines = [
        "# Baseline classifiers summary (script 08)",
        "",
        "## Headline numbers",
        "",
        f"- **Best 3-class LOSO macro-F1**: {best3['macro_f1_mean']:.3f} ± "
        f"{best3['macro_f1_std']:.3f}  (model={best3['model']}, "
        f"preproc={best3['preprocessing']}, mean_acc={best3['mean_acc']:.3f})",
        f"- **Best binary LOSO macro-F1**: {bestb['macro_f1_mean']:.3f} ± "
        f"{bestb['macro_f1_std']:.3f}  (model={bestb['model']}, "
        f"preproc={bestb['preprocessing']}, mean_acc={bestb['mean_acc']:.3f})",
        f"- **Best arm-vs-hand LOSO macro-F1**: "
        f"{besta['macro_f1_mean']:.3f} ± {besta['macro_f1_std']:.3f}  "
        f"(model={besta['model']}, preproc={besta['preprocessing']}, "
        f"mean_acc={besta['mean_acc']:.3f})",
        f"- Validation macro-F1 for best 3-class config: **{val_f1:.3f}**",
        "",
        "Chance-level references: 3-class = 0.333, binary = 0.500, "
        "arm-vs-hand = 0.500.",
        "",
        "## Does subject-z preprocessing help?",
        "",
        "Mean macro-F1 across models by (label_scheme × preprocessing):",
        "",
        ab.round(3).to_markdown(),
        "",
        "## Worst-performing subjects (bottom-5 LOSO accuracy, best 3-class config)",
        "",
        best_pf[["subject_held_out", "accuracy", "macro_f1"]]
        .round(3).to_markdown(index=False),
        "",
        "## Top 15 features (blended importance, 3-class LOSO)",
        "",
        top15.round(2).to_markdown(index=False),
        "",
        "## Generalization gap (LOSO → validation)",
        "",
    ]

    gap = (
        results_df[results_df["split_eval_type"] == "loso_train"]
        [["model", "preprocessing", "label_scheme", "macro_f1_mean"]]
        .merge(val_df[["model", "preprocessing", "label_scheme", "macro_f1"]],
               on=["model", "preprocessing", "label_scheme"])
    )
    gap["gap_loso_minus_val"] = gap["macro_f1_mean"] - gap["macro_f1"]
    lines.append(gap.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("## Modelling recommendations")
    lines.append("")
    recs = [
        "- Non-linear ensembles (RF / XGB) should outperform linear logreg "
        "when subject-z features carry the signal; compare above.",
        "- Subject-z normalisation mitigates the known DC-offset confound — "
        "prefer it over global-z if the table above shows a gain.",
        "- The bottom-5 LOSO subjects are candidates for per-subject "
        "artefact review before any heavier modelling.",
        "- Large LOSO → validation gap means the training subjects are "
        "not representative; consider stratifying feature selection and "
        "using more conservative models.",
        "- For the challenge: ensemble RF+XGB with subject-z features, "
        "tune on LOSO macro-F1; arm-vs-hand requires richer temporal "
        "features than aggregate stats offered here.",
    ]
    lines.extend(recs)
    REPORT_DIR.joinpath("08_baselines_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
