"""Training diagnostics: learning curves, over/underfit check, calibration, scatter plots."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, apply_subject_z, armhand_binary,
    channel_of, class_codes_3, decode_joint_weighted, fit_binary_calibrator,
    fit_binary_proba, load_clean_features, make_binary_model, make_scaler,
    metrics_multiclass, pain_binary, unique_by_canonical, HAS_XGB,
)

OUT = Path("results/final/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)


def xgb_learning_curve(df_norm, feats, stage_name):
    """Fit XGB with eval_set, plot train vs val logloss per round."""
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    y_tr = (train["class"] != "NoPain").astype(int).to_numpy() if stage_name == "stage1" else armhand_binary(train[train["class"].isin(ARM_HAND)]["class"])
    if stage_name == "stage2":
        train = train[train["class"].isin(ARM_HAND)].reset_index(drop=True)
    X_tr = train[feats].to_numpy(dtype=np.float32)
    X_va = val[feats].to_numpy(dtype=np.float32)
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr)
    X_va = sc.transform(X_va)
    y_va = (val["class"] != "NoPain").astype(int).to_numpy() if stage_name == "stage1" else armhand_binary(val[val["class"].isin(ARM_HAND)]["class"])
    if stage_name == "stage2":
        X_va = val[val["class"].isin(ARM_HAND)][feats].to_numpy(dtype=np.float32)
        X_va = sc.transform(X_va)

    from xgboost import XGBClassifier
    mdl = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.08, max_bin=128,
                        tree_method="hist", objective="binary:logistic", eval_metric="logloss",
                        random_state=42, n_jobs=4, verbosity=0)
    mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)
    res = mdl.evals_result()
    train_ll = res["validation_0"]["logloss"]
    val_ll = res["validation_1"]["logloss"]
    rounds = np.arange(1, len(train_ll) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, train_ll, label="train", linewidth=2)
    ax.plot(rounds, val_ll, label="validation", linewidth=2, linestyle="--")
    ax.axvline(200, color="red", linestyle=":", alpha=0.5, label="default n_est=200")
    ax.set_xlabel("boosting round")
    ax.set_ylabel("logloss")
    gap = val_ll[-1] - train_ll[-1]
    ax.set_title(f"XGB {stage_name} learning curve  |  final train={train_ll[-1]:.3f} val={val_ll[-1]:.3f} gap={gap:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"xgb_curve_{stage_name}.png", dpi=140)
    plt.close(fig)

    # Feature importance top 20
    imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp[::-1].plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(imp)))
    ax.set_xlabel("importance")
    ax.set_title(f"Top 20 XGB features ({stage_name})")
    fig.tight_layout()
    fig.savefig(OUT / f"feature_importance_{stage_name}.png", dpi=140)
    plt.close(fig)

    return {"train_final": train_ll[-1], "val_final": val_ll[-1], "gap": gap,
            "optimal_round": int(np.argmin(val_ll)) + 1, "min_val_logloss": min(val_ll)}


def data_size_curve(df_norm, feats, stage_name):
    """Vary training size, plot train/val F1."""
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    if stage_name == "stage2":
        train = train[train["class"].isin(ARM_HAND)].reset_index(drop=True)
        val = val[val["class"].isin(ARM_HAND)].reset_index(drop=True)

    subjects = sorted(train["subject"].unique())
    np.random.seed(42)
    np.random.shuffle(subjects)

    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for frac in fractions:
        n = max(2, int(len(subjects) * frac))
        subset = subjects[:n]
        tr = train[train["subject"].isin(subset)].reset_index(drop=True)
        if stage_name == "stage1":
            y_tr = pain_binary(tr["class"])
            y_va = pain_binary(val["class"])
        else:
            y_tr = armhand_binary(tr["class"])
            y_va = armhand_binary(val["class"])
        if len(np.unique(y_tr)) < 2:
            continue
        spec = ModelSpec("xgb" if stage_name == "stage1" else "logreg", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "C": 1.0})
        probs_tr = fit_binary_proba(tr, tr, feats, y_tr, "std" if stage_name == "stage1" else "robust", spec)
        probs_va = fit_binary_proba(tr, val, feats, y_tr, "std" if stage_name == "stage1" else "robust", spec)
        pred_tr = (probs_tr[:, 1] > 0.5).astype(int)
        pred_va = (probs_va[:, 1] > 0.5).astype(int)
        f1_tr = metrics_multiclass(y_tr, pred_tr)["macro_f1"]
        f1_va = metrics_multiclass(y_va, pred_va)["macro_f1"]
        rows.append({"frac": frac, "n_subj": n, "train_f1": f1_tr, "val_f1": f1_va})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_subj"], df["train_f1"], marker="o", label="train", linewidth=2)
    ax.plot(df["n_subj"], df["val_f1"], marker="s", label="validation", linewidth=2)
    ax.set_xlabel("training subjects")
    ax.set_ylabel("binary macro-F1")
    ax.set_title(f"Data size learning curve ({stage_name})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"data_size_curve_{stage_name}.png", dpi=140)
    plt.close(fig)
    df.to_csv(OUT / f"data_size_curve_{stage_name}.csv", index=False)


def calibration_and_roc(df_norm, feats, stage_name):
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    if stage_name == "stage2":
        train = train[train["class"].isin(ARM_HAND)].reset_index(drop=True)
        val = val[val["class"].isin(ARM_HAND)].reset_index(drop=True)
    if stage_name == "stage1":
        y_tr = pain_binary(train["class"])
        y_va = pain_binary(val["class"])
        spec = ModelSpec("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08})
        scaler_name = "std"
    else:
        y_tr = armhand_binary(train["class"])
        y_va = armhand_binary(val["class"])
        spec = ModelSpec("logreg", {"C": 1.0})
        scaler_name = "robust"
    val_probs = fit_binary_proba(train, val, feats, y_tr, scaler_name, spec)
    prob_pos = val_probs[:, 1]
    # Calibration curve
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    frac_pos, mean_pred = calibration_curve(y_va, prob_pos, n_bins=10, strategy="quantile")
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[0].plot(mean_pred, frac_pos, marker="o", linewidth=2)
    ax[0].set_xlabel("predicted probability")
    ax[0].set_ylabel("observed frequency")
    ax[0].set_title(f"Reliability diagram ({stage_name})")
    ax[0].grid(alpha=0.3)
    # ROC
    fpr, tpr, _ = roc_curve(y_va, prob_pos)
    auc = roc_auc_score(y_va, prob_pos)
    ax[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[1].plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.3f}")
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    ax[1].set_title(f"ROC ({stage_name})")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"calibration_roc_{stage_name}.png", dpi=140)
    plt.close(fig)
    return auc


def loso_vs_val_scatter():
    """Read existing per-subject CSVs, plot LOSO mean vs VAL on same chart."""
    loso = pd.read_csv("results/final/final_pipeline/loso_per_subject.csv")
    val = pd.read_csv("results/final/final_pipeline/val_per_subject.csv")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(loso["f1"], bins=15, alpha=0.7, label=f"LOSO (n={len(loso)})", color="steelblue")
    ax[0].hist(val["f1"], bins=15, alpha=0.7, label=f"VAL (n={len(val)})", color="coral")
    ax[0].axvline(loso["f1"].mean(), color="blue", linestyle="--", label=f"LOSO mean={loso['f1'].mean():.3f}")
    ax[0].axvline(val["f1"].mean(), color="red", linestyle="--", label=f"VAL mean={val['f1'].mean():.3f}")
    ax[0].axvline(0.5, color="black", linestyle=":", alpha=0.5, label="F1=0.5")
    ax[0].set_xlabel("F1")
    ax[0].set_ylabel("count")
    ax[0].set_title("Per-subject F1 distribution: LOSO vs VAL")
    ax[0].legend()
    # Sorted per-subject
    ax[1].plot(np.sort(loso["f1"])[::-1], marker="o", label=f"LOSO sorted (n={len(loso)})", color="steelblue")
    ax[1].plot(np.linspace(0, len(loso) - 1, len(val)), np.sort(val["f1"])[::-1], marker="s", label=f"VAL sorted (n={len(val)})", color="coral")
    ax[1].axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax[1].set_xlabel("rank")
    ax[1].set_ylabel("F1")
    ax[1].set_title("Sorted per-subject F1")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "loso_vs_val_distribution.png", dpi=140)
    plt.close(fig)


def main() -> None:
    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    resp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "resp"])
    s1_feats = bvp + eda
    s2_feats = resp

    s1_df = apply_subject_robust(df_all, feat_cols)
    s2_df = apply_subject_z(df_all, feat_cols)

    results = {}
    if HAS_XGB:
        print(">>> XGB stage 1 learning curve...")
        results["stage1_xgb"] = xgb_learning_curve(s1_df, s1_feats, "stage1")
        print(f"  {results['stage1_xgb']}")
        print(">>> XGB stage 2 learning curve...")
        results["stage2_xgb"] = xgb_learning_curve(s2_df, s2_feats, "stage2")
        print(f"  {results['stage2_xgb']}")

    print(">>> Data size curves...")
    data_size_curve(s1_df, s1_feats, "stage1")
    data_size_curve(s2_df, s2_feats, "stage2")

    print(">>> Calibration + ROC...")
    auc_s1 = calibration_and_roc(s1_df, s1_feats, "stage1")
    auc_s2 = calibration_and_roc(s2_df, s2_feats, "stage2")
    print(f"  stage1 AUC={auc_s1:.3f}  stage2 AUC={auc_s2:.3f}")

    print(">>> LOSO vs VAL distribution...")
    if Path("results/final/final_pipeline/loso_per_subject.csv").exists():
        loso_vs_val_scatter()

    pd.Series({**{f"{k}_{kk}": vv for k, v in results.items() for kk, vv in v.items()},
               "stage1_auc": auc_s1, "stage2_auc": auc_s2}).to_csv(OUT / "summary.csv")
    print(f"\nAll plots saved: {OUT}")


if __name__ == "__main__":
    main()
