"""Per-modality tables for both stages + combined + training logs + ROC + confusion."""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score,
)
from tqdm import tqdm

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, apply_subject_z, armhand_binary,
    channel_of, class_codes_3, decode_joint_weighted, fit_binary_calibrator,
    fit_binary_proba, load_clean_features, make_scaler,
    unique_by_canonical,
)

OUT = Path("results/final/modality_tables")
OUT.mkdir(parents=True, exist_ok=True)

MODALITIES = {
    "BVP": ["bvp"],
    "EDA": ["eda"],
    "RESP": ["resp"],
    "SpO2": ["spo2"],
    "EDA+BVP": ["eda", "bvp"],
    "EDA+BVP+RESP": ["eda", "bvp", "resp"],
    "All": ["bvp", "eda", "resp", "spo2"],
}


def select(feat_cols, tags):
    return unique_by_canonical([c for c in feat_cols if channel_of(c) in tags])


def exact12_nopain(df_sub, score_pain):
    """Predict exact 12 NoPain per subject using lowest pain-prob segments."""
    pred = np.ones(len(df_sub), dtype=int)
    for s in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == s).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(score_pain[mask])]  # ascending = lowest pain = NoPain first
        nop_set = set(order[:12].tolist())
        pred[mask] = [0 if i in nop_set else 1 for i in idx]
    return pred


def exact12_arm(df_sub, arm_score):
    """For pain-only val rows: exact 12 arm per subject."""
    pred = np.zeros(len(df_sub), dtype=int)
    for s in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == s).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-arm_score[mask])]
        arm_set = set(order[:12].tolist())
        pred[mask] = [1 if i in arm_set else 0 for i in idx]  # 1=Arm, 0=Hand
    return pred


def fit_stage1(df_norm, feats):
    spec = ModelSpec("xgb", {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    tp = np.zeros((len(train), 2), dtype=np.float32)
    for s in sorted(train["subject"].unique()):
        tr = train[train["subject"] != s].reset_index(drop=True)
        te = train[train["subject"] == s].reset_index(drop=True)
        tp[train["subject"] == s] = fit_binary_proba(tr, te, feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    vp = fit_binary_proba(train, val, feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    # Calibrator maps raw P(NoPain) → calibrated P(NoPain)
    cal = fit_binary_calibrator("sigmoid", tp[:, 0], (train["class"] == "NoPain").astype(int).to_numpy())
    nopain_prob_tr = cal(tp[:, 0]).astype(np.float32)
    nopain_prob_va = cal(vp[:, 0]).astype(np.float32)
    pain_prob_tr = 1.0 - nopain_prob_tr
    pain_prob_va = 1.0 - nopain_prob_va
    return train, val, pain_prob_tr, pain_prob_va


def fit_stage2(df_norm, feats):
    spec = ModelSpec("logreg", {"C": 1.0})
    train_full = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val_full = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    pain_train = train_full[train_full["class"].isin(ARM_HAND)].reset_index(drop=True)
    tp = np.zeros((len(train_full), 2), dtype=np.float32)
    for s in sorted(train_full["subject"].unique()):
        pain_tr = pain_train[pain_train["subject"] != s].reset_index(drop=True)
        te = train_full[train_full["subject"] == s].reset_index(drop=True)
        if len(pain_tr) == 0:
            continue
        tp[train_full["subject"] == s] = fit_binary_proba(pain_tr, te, feats, armhand_binary(pain_tr["class"]), "robust", spec)
    vp = fit_binary_proba(pain_train, val_full, feats, armhand_binary(pain_train["class"]), "robust", spec)
    pm = (train_full["class"].isin(ARM_HAND)).to_numpy()
    y = (train_full.loc[pm, "class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator("isotonic", tp[pm, 0], y)
    return train_full, val_full, cal(tp[:, 0]).astype(np.float32), cal(vp[:, 0]).astype(np.float32)


def xgb_curve(df_norm, feats, n_est=300):
    from xgboost import XGBClassifier
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    y_tr = (train["class"] != "NoPain").astype(int).to_numpy()
    y_va = (val["class"] != "NoPain").astype(int).to_numpy()
    sc = make_scaler("std").fit(train[feats].to_numpy(dtype=np.float32))
    Xtr = sc.transform(train[feats].to_numpy(dtype=np.float32))
    Xva = sc.transform(val[feats].to_numpy(dtype=np.float32))
    mdl = XGBClassifier(n_estimators=n_est, max_depth=4, learning_rate=0.08, max_bin=128,
                        tree_method="hist", objective="binary:logistic", eval_metric="logloss",
                        random_state=42, n_jobs=4, verbosity=0)
    mdl.fit(Xtr, y_tr, eval_set=[(Xtr, y_tr), (Xva, y_va)], verbose=False)
    ev = mdl.evals_result()
    return pd.DataFrame({
        "iter": np.arange(1, len(ev["validation_0"]["logloss"]) + 1),
        "train_logloss": ev["validation_0"]["logloss"],
        "val_logloss": ev["validation_1"]["logloss"],
    })


def plot_cm(cm, names, title, path):
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_curve(hist, title, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["iter"], hist["train_logloss"], label="train", linewidth=2)
    ax.plot(hist["iter"], hist["val_logloss"], label="validation", linewidth=2, linestyle="--")
    ax.set_xlabel("boosting round"); ax.set_ylabel("logloss")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_roc_multi(roc_data, title, path):
    """roc_data = dict of modality -> (fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def classwise(y_true, y_pred, labels, names):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, n in enumerate(names)}


def main():
    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    s1_df = apply_subject_robust(df_all, feat_cols)
    s2_df = apply_subject_z(df_all, feat_cols)

    stage1_rows, stage2_rows, combined_rows = [], [], []
    roc_s1, roc_s2 = {}, {}

    for mod, tags in tqdm(MODALITIES.items(), desc="modality"):
        feats = select(feat_cols, tags)
        if len(feats) < 3:
            continue

        # STAGE 1
        train_s1, val_s1, pain_t, pain_v = fit_stage1(s1_df, feats)
        y_true = (val_s1["class"] != "NoPain").astype(int).to_numpy()
        y_pred = exact12_nopain(val_s1, pain_v)
        auc_s1 = roc_auc_score(y_true, pain_v)
        acc_s1 = accuracy_score(y_true, y_pred)
        cm_s1 = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cw_s1 = classwise(y_true, y_pred, [0, 1], ["NoPain", "Pain"])
        stage1_rows.append({
            "modality": mod, "n_features": len(feats),
            "val_acc": acc_s1, "val_auc": auc_s1,
            **{f"{n}_{m}": v for n, mets in cw_s1.items() for m, v in mets.items()},
        })
        plot_cm(cm_s1, ["NoPain", "Pain"], f"Stage 1 {mod} (AUC={auc_s1:.3f})", OUT / f"cm_stage1_{mod}.png")
        hist = xgb_curve(s1_df, feats, 300)
        hist.to_csv(OUT / f"curve_stage1_{mod}.csv", index=False)
        plot_curve(hist, f"Stage 1 {mod} XGB learning", OUT / f"curve_stage1_{mod}.png")
        fpr, tpr, _ = roc_curve(y_true, pain_v)
        roc_s1[mod] = (fpr, tpr, auc_s1)

        # STAGE 2
        train_s2, val_s2, arm_t, arm_v = fit_stage2(s2_df, feats)
        pm = val_s2["class"].isin(ARM_HAND).to_numpy()
        val_pain = val_s2[pm].reset_index(drop=True)
        arm_v_pain = arm_v[pm]
        y_true_s2 = (val_pain["class"] == "PainArm").astype(int).to_numpy()
        y_pred_s2 = exact12_arm(val_pain, arm_v_pain)
        auc_s2 = roc_auc_score(y_true_s2, arm_v_pain)
        acc_s2 = accuracy_score(y_true_s2, y_pred_s2)
        cm_s2 = confusion_matrix(y_true_s2, y_pred_s2, labels=[1, 0])
        cw_s2 = classwise(y_true_s2, y_pred_s2, [1, 0], ["PainArm", "PainHand"])
        stage2_rows.append({
            "modality": mod, "n_features": len(feats),
            "val_acc": acc_s2, "val_auc": auc_s2,
            **{f"{n}_{m}": v for n, mets in cw_s2.items() for m, v in mets.items()},
        })
        plot_cm(cm_s2, ["PainArm", "PainHand"], f"Stage 2 {mod} (AUC={auc_s2:.3f})", OUT / f"cm_stage2_{mod}.png")
        fpr2, tpr2, _ = roc_curve(y_true_s2, arm_v_pain)
        roc_s2[mod] = (fpr2, tpr2, auc_s2)

        # COMBINED 3-class
        mp = dict(zip(val_s2["segment_id"], arm_v))
        arm_v_aligned = np.array([mp[sid] for sid in val_s1["segment_id"]], dtype=np.float32)
        s1_probs = np.column_stack([1 - pain_v, pain_v]).astype(np.float32)
        s2_probs = np.column_stack([arm_v_aligned, 1 - arm_v_aligned]).astype(np.float32)
        y_pred_3 = np.zeros(len(val_s1), dtype=int)
        for subj in sorted(val_s1["subject"].unique()):
            mask = (val_s1["subject"] == subj).to_numpy()
            y_pred_3[mask] = decode_joint_weighted(s1_probs[mask], s2_probs[mask], 1, 1, 1)
        y_true_3 = class_codes_3(val_s1["class"])
        cm_3 = confusion_matrix(y_true_3, y_pred_3, labels=[0, 1, 2])
        cw_3 = classwise(y_true_3, y_pred_3, [0, 1, 2], ["NoPain", "PainArm", "PainHand"])
        acc_3 = accuracy_score(y_true_3, y_pred_3)
        macro_f1 = np.mean([cw_3[n]["f1"] for n in ["NoPain", "PainArm", "PainHand"]])
        combined_rows.append({
            "modality": mod, "n_features": len(feats),
            "val_acc": acc_3, "val_macro_f1": macro_f1,
            **{f"{n}_{m}": v for n, mets in cw_3.items() for m, v in mets.items()},
        })
        plot_cm(cm_3, ["NoPain", "Arm", "Hand"], f"Combined {mod} (acc={acc_3:.3f})", OUT / f"cm_combined_{mod}.png")

    # Save tables
    pd.DataFrame(stage1_rows).to_csv(OUT / "stage1_modality_table.csv", index=False)
    pd.DataFrame(stage2_rows).to_csv(OUT / "stage2_modality_table.csv", index=False)
    pd.DataFrame(combined_rows).to_csv(OUT / "combined_modality_table.csv", index=False)

    # ROC multi-plot
    plot_roc_multi(roc_s1, "Stage 1 ROC per modality (NoPain vs Pain)", OUT / "roc_stage1.png")
    plot_roc_multi(roc_s2, "Stage 2 ROC per modality (Arm vs Hand)", OUT / "roc_stage2.png")

    # Markdown
    def cw_table(rows, classes):
        hdr = "| Modality | " + " | ".join([f"{c} P" for c in classes] + [f"{c} R" for c in classes] + [f"{c} F1" for c in classes]) + " |"
        sep = "|---" + "|---:" * (3 * len(classes)) + "|"
        lines = [hdr, sep]
        for r in rows:
            vals = [f"{r[f'{c}_precision']:.3f}" for c in classes] + [f"{r[f'{c}_recall']:.3f}" for c in classes] + [f"{r[f'{c}_f1']:.3f}" for c in classes]
            lines.append(f"| {r['modality']} | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def overall_table(rows, cols):
        hdr = "| Modality | n_features | " + " | ".join(cols) + " |"
        sep = "|---|---:|" + "---:|" * len(cols)
        lines = [hdr, sep]
        for r in rows:
            vals = [f"{r[c]:.3f}" for c in cols]
            lines.append(f"| {r['modality']} | {r['n_features']} | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    rpt = [
        "# Modality Performance Tables",
        "",
        "## Stage 1 (NoPain vs Pain) — Classwise", "",
        cw_table(stage1_rows, ["NoPain", "Pain"]), "",
        "## Stage 1 Overall", "",
        overall_table(stage1_rows, ["val_acc", "val_auc"]), "",
        "## Stage 2 (PainArm vs PainHand) — Classwise", "",
        cw_table(stage2_rows, ["PainArm", "PainHand"]), "",
        "## Stage 2 Overall", "",
        overall_table(stage2_rows, ["val_acc", "val_auc"]), "",
        "## Combined 3-Class — Classwise", "",
        cw_table(combined_rows, ["NoPain", "PainArm", "PainHand"]), "",
        "## Combined Overall", "",
        overall_table(combined_rows, ["val_acc", "val_macro_f1"]),
    ]
    (OUT / "report.md").write_text("\n".join(rpt))
    print("\n".join(rpt))


if __name__ == "__main__":
    main()
