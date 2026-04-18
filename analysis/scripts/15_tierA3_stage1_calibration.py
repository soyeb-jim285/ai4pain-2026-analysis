"""Tier-A3 — Stage-1 (pain vs no-pain) calibration & ARM/HAND drift analysis.

Question: does the stage-1 binary classifier's pain-probability systematically
differ between PainArm and PainHand segments?  If so, that probability is a
"free" meta-feature for stage-2 (arm vs hand) localisation.

Pipeline
--------
1. Recompute stage-1 LOSO with predict_proba (XGBoost, subject-z features).
2. Per-subject paired Wilcoxon: mean pain_prob | ARM vs mean pain_prob | HAND.
3. Pearson + point-biserial correlation of pain_prob with ARM/HAND label
   inside pain-only segments (global + per subject).
4. Train logistic regression LOSO on {PainArm, PainHand} using ONLY pain_prob
   as a single feature, to quantify the meta-feature's stand-alone utility.
5. Plots + markdown report.

Outputs
-------
- results/tables/tierA3_stage1_probabilities.csv
- results/tables/tierA3_stage1_armhand_drift.csv
- results/tables/tierA3_stage1_correlation.csv
- results/tables/tierA3_stage1_as_feature_loso.csv
- plots/tierA3_stage1/{pain_prob_by_class_violin,
                       pain_prob_arm_minus_hand_per_subject,
                       pain_prob_distribution_arm_vs_hand,
                       pain_prob_calibration_curve}.png
- results/reports/15_tierA3_stage1_calibration_summary.md

Usage
-----
    uv run python scripts/15_tierA3_stage1_calibration.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

# Non-interactive matplotlib backend (XGBoost workers can otherwise grab Tk).
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sstats  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from tqdm import tqdm  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ANALYSIS = Path(__file__).resolve().parents[1]
TAB_DIR = ANALYSIS / "results" / "tables"
REPORT_DIR = ANALYSIS / "results" / "reports"
PLOT_DIR = ANALYSIS / "plots" / "tierA3_stage1"
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
FEATURE_PARQUET = TAB_DIR / "all_features_merged.parquet"


# ---------------------------------------------------------------------------
# Feature prep
# ---------------------------------------------------------------------------
def load_and_filter_features() -> tuple[pd.DataFrame, list[str]]:
    """Load merged features, drop high-NaN / zero-var, median-impute."""
    df = pd.read_parquet(FEATURE_PARQUET).reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    nan_frac = df[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if nan_frac[c] <= 0.10]
    X = df[feat_cols].astype(np.float64)
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=feat_cols,
    )
    var = X.var(axis=0)
    keep = var.index[var > 1e-12].tolist()
    X = X[keep]
    out = pd.concat(
        [df[META_COLS].reset_index(drop=True),
         X.astype(np.float32).reset_index(drop=True)],
        axis=1,
    )
    return out, keep


def apply_subject_z(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Per-subject z-score (mean 0 / sd 1 within each subject)."""
    out = df.copy()
    means = out.groupby("subject")[feat_cols].transform("mean")
    stds = out.groupby("subject")[feat_cols].transform("std", ddof=0)
    stds = stds.where(stds > 0, 1.0)
    out[feat_cols] = (out[feat_cols] - means) / stds
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        max_bin=128,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=4,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Stage-1 LOSO with probabilities
# ---------------------------------------------------------------------------
def compute_stage1_probabilities(
    df_z: pd.DataFrame, feat_cols: list[str]
) -> pd.DataFrame:
    """Returns long DataFrame with one row per segment (train LOSO + val)."""
    train = df_z[df_z["split"] == "train"].reset_index(drop=True)
    val = df_z[df_z["split"] == "validation"].reset_index(drop=True)

    y_tr_full = (train["class"] != "NoPain").astype(int).to_numpy()
    X_tr_full = train[feat_cols].to_numpy()
    subjects_tr = train["subject"].to_numpy()

    rows: list[dict] = []

    # ---- LOSO on train ----
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_tr_full, y_tr_full, groups=subjects_tr))
    for tr_idx, te_idx in tqdm(folds, desc="stage-1 LOSO", leave=False):
        mdl = make_xgb()
        mdl.fit(X_tr_full[tr_idx], y_tr_full[tr_idx])
        proba = mdl.predict_proba(X_tr_full[te_idx])[:, 1]  # P(pain)
        held = train.iloc[te_idx]
        for (_, row), p in zip(held.iterrows(), proba):
            rows.append({
                "subject": int(row["subject"]),
                "segment_id": row["segment_id"],
                "class": row["class"],
                "split": "train",
                "pain_prob": float(p),
                "is_pain_true": int(row["class"] != "NoPain"),
            })

    # ---- Validation: train on full train, score val ----
    if len(val):
        mdl = make_xgb()
        mdl.fit(X_tr_full, y_tr_full)
        Xv = val[feat_cols].to_numpy()
        proba_v = mdl.predict_proba(Xv)[:, 1]
        for (_, row), p in zip(val.iterrows(), proba_v):
            rows.append({
                "subject": int(row["subject"]),
                "segment_id": row["segment_id"],
                "class": row["class"],
                "split": "validation",
                "pain_prob": float(p),
                "is_pain_true": int(row["class"] != "NoPain"),
            })

    out = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# ARM vs HAND drift on stage-1 probabilities
# ---------------------------------------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta for two independent samples a and b. Range [-1, 1]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    n_gt = float((diff > 0).sum())
    n_lt = float((diff < 0).sum())
    return (n_gt - n_lt) / (a.size * b.size)


def armhand_drift(probs: pd.DataFrame, split: str) -> dict:
    """Per-subject paired Wilcoxon on mean pain_prob | ARM vs | HAND."""
    sub_df = probs[
        (probs["split"] == split)
        & (probs["class"].isin(["PainArm", "PainHand"]))
    ]
    pivot = (
        sub_df.groupby(["subject", "class"])["pain_prob"].mean().unstack("class")
    )
    pivot = pivot.dropna(subset=["PainArm", "PainHand"])

    arm = pivot["PainArm"].to_numpy()
    hand = pivot["PainHand"].to_numpy()
    diff = arm - hand
    n = len(diff)

    if n < 2 or np.all(diff == 0):
        return {
            "split": split, "n_subjects": n,
            "mean_arm": float(np.mean(arm)) if n else float("nan"),
            "mean_hand": float(np.mean(hand)) if n else float("nan"),
            "mean_diff_arm_minus_hand": float(np.mean(diff)) if n else float("nan"),
            "wilcoxon_W": float("nan"),
            "wilcoxon_p": float("nan"),
            "ttest_t": float("nan"),
            "ttest_p": float("nan"),
            "cliffs_delta_arm_vs_hand": float("nan"),
            "sign_consistency_arm_gt_hand": float("nan"),
        }

    try:
        w_stat, w_p = sstats.wilcoxon(arm, hand, zero_method="wilcox",
                                      alternative="two-sided")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    t_stat, t_p = sstats.ttest_rel(arm, hand)

    # Cliff's delta on the segment-level distributions, not subject means
    arm_segs = sub_df.loc[sub_df["class"] == "PainArm", "pain_prob"].to_numpy()
    hand_segs = sub_df.loc[sub_df["class"] == "PainHand", "pain_prob"].to_numpy()
    delta = cliffs_delta(arm_segs, hand_segs)

    sign_consistency = float((diff > 0).sum()) / n  # fraction with ARM > HAND

    return {
        "split": split,
        "n_subjects": n,
        "mean_arm": float(np.mean(arm)),
        "mean_hand": float(np.mean(hand)),
        "mean_diff_arm_minus_hand": float(np.mean(diff)),
        "std_diff_arm_minus_hand": float(np.std(diff, ddof=1)),
        "wilcoxon_W": float(w_stat),
        "wilcoxon_p": float(w_p),
        "ttest_t": float(t_stat),
        "ttest_p": float(t_p),
        "cliffs_delta_arm_vs_hand": float(delta),
        "sign_consistency_arm_gt_hand": sign_consistency,
    }


def per_segment_correlations(probs: pd.DataFrame) -> pd.DataFrame:
    """Pearson + point-biserial of pain_prob ~ class (ARM=0, HAND=1)
    within pain segments only.  Reported globally and per subject."""
    pain = probs[probs["class"].isin(["PainArm", "PainHand"])].copy()
    pain["label"] = (pain["class"] == "PainHand").astype(int)

    rows: list[dict] = []
    for split in ["train", "validation"]:
        sub = pain[pain["split"] == split]
        if len(sub) < 3:
            continue
        x = sub["pain_prob"].to_numpy()
        y = sub["label"].to_numpy()
        if np.std(x) == 0 or len(np.unique(y)) < 2:
            r_p, p_p = float("nan"), float("nan")
            r_pb, p_pb = float("nan"), float("nan")
        else:
            r_p, p_p = sstats.pearsonr(x, y)
            r_pb, p_pb = sstats.pointbiserialr(y, x)
        rows.append({
            "scope": "global",
            "split": split,
            "subject": "ALL",
            "n": int(len(sub)),
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "pointbiserial_r": float(r_pb),
            "pointbiserial_p": float(p_pb),
        })

        for sid, g in sub.groupby("subject"):
            if len(g) < 3 or g["label"].nunique() < 2 or np.std(g["pain_prob"]) == 0:
                rows.append({
                    "scope": "per_subject", "split": split,
                    "subject": int(sid), "n": int(len(g)),
                    "pearson_r": float("nan"), "pearson_p": float("nan"),
                    "pointbiserial_r": float("nan"), "pointbiserial_p": float("nan"),
                })
                continue
            r1, p1 = sstats.pearsonr(g["pain_prob"], g["label"])
            r2, p2 = sstats.pointbiserialr(g["label"], g["pain_prob"])
            rows.append({
                "scope": "per_subject", "split": split,
                "subject": int(sid), "n": int(len(g)),
                "pearson_r": float(r1), "pearson_p": float(p1),
                "pointbiserial_r": float(r2), "pointbiserial_p": float(p2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# pain_prob alone as ARM vs HAND classifier
# ---------------------------------------------------------------------------
def pain_prob_as_feature_loso(probs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    train_pain = probs[
        (probs["split"] == "train")
        & (probs["class"].isin(["PainArm", "PainHand"]))
    ].reset_index(drop=True)

    X = train_pain[["pain_prob"]].to_numpy()
    y = (train_pain["class"] == "PainHand").astype(int).to_numpy()  # HAND=1
    groups = train_pain["subject"].to_numpy()

    logo = LeaveOneGroupOut()
    per_fold = []
    for fold, (tr, te) in enumerate(
        tqdm(list(logo.split(X, y, groups=groups)),
             desc="pain_prob LOSO", leave=False)
    ):
        mdl = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            max_iter=2000, solver="lbfgs", random_state=SEED, n_jobs=1,
        )
        mdl.fit(X[tr], y[tr])
        yhat = mdl.predict(X[te])
        per_fold.append({
            "fold": fold,
            "held_out_subject": int(groups[te][0]),
            "n_test": int(len(te)),
            "macro_f1": f1_score(y[te], yhat, average="macro", zero_division=0),
            "balanced_acc": balanced_accuracy_score(y[te], yhat),
        })
    df_fold = pd.DataFrame(per_fold)
    summary = {
        "n_folds": int(len(df_fold)),
        "macro_f1_mean": float(df_fold["macro_f1"].mean()),
        "macro_f1_std": float(df_fold["macro_f1"].std(ddof=0)),
        "balanced_acc_mean": float(df_fold["balanced_acc"].mean()),
        "balanced_acc_std": float(df_fold["balanced_acc"].std(ddof=0)),
        "chance": 0.50,
    }
    return summary, df_fold


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_violin_by_class(probs: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = probs[probs["split"] == "train"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=sub, x="class", y="pain_prob",
        order=["NoPain", "PainArm", "PainHand"],
        inner="quartile", cut=0, ax=ax,
        palette=["#a6cee3", "#fb9a99", "#e31a1c"],
    )
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.6)
    ax.set_ylabel("stage-1 P(pain)")
    ax.set_xlabel("")
    ax.set_title("Stage-1 pain probability by class (train, LOSO)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pain_prob_by_class_violin.png", dpi=130)
    plt.close(fig)


def plot_per_subject_diff(probs: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    sub = probs[
        (probs["split"] == "train")
        & (probs["class"].isin(["PainArm", "PainHand"]))
    ]
    pivot = (
        sub.groupby(["subject", "class"])["pain_prob"].mean().unstack("class")
    )
    pivot["diff"] = pivot["PainArm"] - pivot["PainHand"]
    pivot = pivot.sort_values("diff")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    colors = ["#1f77b4" if d >= 0 else "#d62728" for d in pivot["diff"]]
    ax.bar(pivot.index.astype(str), pivot["diff"], color=colors,
           edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("mean P(pain | ARM) - mean P(pain | HAND)")
    ax.set_xlabel("subject (train)")
    ax.set_title("Per-subject ARM minus HAND mean stage-1 pain probability")
    plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pain_prob_arm_minus_hand_per_subject.png", dpi=130)
    plt.close(fig)


def plot_kde_arm_vs_hand(probs: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = probs[
        (probs["split"] == "train")
        & (probs["class"].isin(["PainArm", "PainHand"]))
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(
        data=sub, x="pain_prob", hue="class",
        common_norm=False, fill=True, alpha=0.35,
        palette={"PainArm": "#fb9a99", "PainHand": "#e31a1c"},
        ax=ax, clip=(0, 1),
    )
    ax.set_xlabel("stage-1 P(pain)")
    ax.set_title("Distribution of P(pain) for ARM vs HAND segments (train)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pain_prob_distribution_arm_vs_hand.png", dpi=130)
    plt.close(fig)


def plot_calibration(probs: pd.DataFrame) -> None:
    """Reliability plot: bin pain_prob into deciles, compute empirical pain rate.
    Curves coloured by class membership (ARM vs HAND vs NoPain)."""
    import matplotlib.pyplot as plt

    sub = probs[probs["split"] == "train"].copy()
    bins = np.linspace(0.0, 1.0, 11)
    sub["bin"] = pd.cut(sub["pain_prob"], bins=bins,
                        include_lowest=True, labels=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="perfect")

    # Overall (binary): empirical pain rate per bin
    def empirical(df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        agg = df_in.groupby("bin").agg(
            mean_pred=("pain_prob", "mean"),
            actual_pain_rate=("is_pain_true", "mean"),
            n=("pain_prob", "size"),
        ).dropna()
        return agg["mean_pred"].to_numpy(), agg["actual_pain_rate"].to_numpy()

    mp, ap = empirical(sub)
    ax.plot(mp, ap, "o-", color="#333333", linewidth=2,
            label="all segments (binary actual pain)")

    # ARM-only & HAND-only: average true=1 (because pain) but track mean(pred)
    for cls, color in (("PainArm", "#fb9a99"), ("PainHand", "#e31a1c")):
        cs = sub[sub["class"] == cls]
        if not len(cs):
            continue
        mp_c, ap_c = empirical(cs)
        ax.plot(mp_c, ap_c, "s-", color=color, alpha=0.85,
                label=f"{cls} only")

    ax.set_xlabel("mean predicted P(pain) per decile")
    ax.set_ylabel("empirical pain rate per decile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Stage-1 reliability curve (train, LOSO)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pain_prob_calibration_curve.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(
    drift_train: dict, drift_val: dict,
    corr_df: pd.DataFrame, feat_summary: dict,
) -> str:
    lines: list[str] = []
    lines.append("# Tier-A3 — Stage-1 calibration & ARM/HAND drift\n")
    lines.append(
        "Tests whether the stage-1 (pain vs no-pain) classifier's pain "
        "probability differs systematically between PainArm and PainHand "
        "segments, in which case it is a free meta-feature for stage-2.\n"
    )

    lines.append("## 1. Per-subject ARM vs HAND drift (paired Wilcoxon)\n")
    lines.append(
        "| split | n_subj | mean(ARM) | mean(HAND) | Δ (ARM-HAND) | "
        "Wilcoxon W | p | t-test p | Cliff's δ | sign consistency (ARM>HAND) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in (drift_train, drift_val):
        lines.append(
            f"| {d['split']} | {d['n_subjects']} | "
            f"{d['mean_arm']:.3f} | {d['mean_hand']:.3f} | "
            f"{d['mean_diff_arm_minus_hand']:+.4f} | "
            f"{d['wilcoxon_W']:.1f} | {d['wilcoxon_p']:.3g} | "
            f"{d['ttest_p']:.3g} | "
            f"{d['cliffs_delta_arm_vs_hand']:+.3f} | "
            f"{d['sign_consistency_arm_gt_hand']:.2f} |"
        )
    lines.append("")

    lines.append("## 2. Correlation of pain_prob with ARM(0)/HAND(1) "
                 "(pain segments only)\n")
    glob = corr_df[corr_df["scope"] == "global"]
    lines.append("| split | n | Pearson r | p | point-biserial r | p |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in glob.iterrows():
        lines.append(
            f"| {r['split']} | {int(r['n'])} | "
            f"{r['pearson_r']:+.4f} | {r['pearson_p']:.3g} | "
            f"{r['pointbiserial_r']:+.4f} | {r['pointbiserial_p']:.3g} |"
        )
    lines.append("")

    per_subj = corr_df[(corr_df["scope"] == "per_subject")
                       & (corr_df["split"] == "train")]
    if len(per_subj):
        sig = (per_subj["pointbiserial_p"] < 0.05).sum()
        lines.append(
            f"- Per-subject (train): "
            f"median |point-biserial r| = "
            f"{per_subj['pointbiserial_r'].abs().median():.3f}; "
            f"{sig}/{len(per_subj)} subjects reach p<0.05 individually.\n"
        )

    lines.append("## 3. pain_prob alone as ARM vs HAND classifier (LOSO)\n")
    lines.append(
        f"- macro-F1 = **{feat_summary['macro_f1_mean']:.3f} ± "
        f"{feat_summary['macro_f1_std']:.3f}** (chance = 0.50)"
    )
    lines.append(
        f"- balanced acc = **{feat_summary['balanced_acc_mean']:.3f} ± "
        f"{feat_summary['balanced_acc_std']:.3f}**"
    )
    lines.append(f"- n_folds = {feat_summary['n_folds']}\n")

    # ---- recommendation ----
    p_train = drift_train["wilcoxon_p"]
    delta = drift_train["cliffs_delta_arm_vs_hand"]
    macro_f1 = feat_summary["macro_f1_mean"]
    pb_global = float(glob.loc[glob["split"] == "train",
                               "pointbiserial_r"].iloc[0])

    useful = (
        (not np.isnan(p_train) and p_train < 0.05)
        and (abs(delta) >= 0.10)
    ) or (macro_f1 >= 0.55)
    weak = (
        (not np.isnan(p_train) and 0.05 <= p_train < 0.20)
        and (0.05 <= abs(delta) < 0.10)
    ) and (macro_f1 < 0.55)

    lines.append("## 4. Recommendation\n")
    if useful:
        verdict = (
            "**Include pain_prob as a stage-2 feature.** The stage-1 "
            "probability differs significantly between ARM and HAND "
            "(paired Wilcoxon p={:.3g}, Cliff's δ={:+.3f}, point-biserial "
            "r={:+.3f}) and on its own beats chance for ARM vs HAND "
            "discrimination (macro-F1 = {:.3f}).".format(
                p_train, delta, pb_global, macro_f1)
        )
    elif weak:
        verdict = (
            "**Marginal — include only as one of many features.** The "
            "stage-1 probability shows a hint of ARM/HAND drift "
            "(p={:.3g}, δ={:+.3f}) but on its own gives macro-F1 = {:.3f} "
            "which barely exceeds chance.".format(p_train, delta, macro_f1)
        )
    else:
        verdict = (
            "**Skip pain_prob as a meta-feature.** The stage-1 probability "
            "does not differ systematically between ARM and HAND "
            "(p={:.3g}, Cliff's δ={:+.3f}); stand-alone macro-F1 = {:.3f} "
            "is at or below chance.".format(p_train, delta, macro_f1)
        )
    lines.append(verdict + "\n")

    lines.append("## Files\n")
    lines.append("- `results/tables/tierA3_stage1_probabilities.csv`")
    lines.append("- `results/tables/tierA3_stage1_armhand_drift.csv`")
    lines.append("- `results/tables/tierA3_stage1_correlation.csv`")
    lines.append("- `results/tables/tierA3_stage1_as_feature_loso.csv`")
    lines.append("- `plots/tierA3_stage1/*.png`")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("[load] features ...")
    df, feat_cols = load_and_filter_features()
    print(f"[features] {len(feat_cols)} features after NaN/var filter "
          f"× {len(df)} segments")

    df_z = apply_subject_z(df, feat_cols)

    print("[stage-1] LOSO XGBoost predict_proba ...")
    probs = compute_stage1_probabilities(df_z, feat_cols)
    probs_path = TAB_DIR / "tierA3_stage1_probabilities.csv"
    probs.to_csv(probs_path, index=False)
    print(f"[save] {probs_path}")

    # ARM vs HAND drift
    print("[drift] paired tests on ARM vs HAND mean pain_prob ...")
    drift_train = armhand_drift(probs, "train")
    drift_val = armhand_drift(probs, "validation")
    drift_df = pd.DataFrame([drift_train, drift_val])
    drift_path = TAB_DIR / "tierA3_stage1_armhand_drift.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"[save] {drift_path}")

    # Correlations
    print("[corr] Pearson + point-biserial of pain_prob vs class ...")
    corr_df = per_segment_correlations(probs)
    corr_path = TAB_DIR / "tierA3_stage1_correlation.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"[save] {corr_path}")

    # pain_prob alone as classifier
    print("[feat] pain_prob alone as ARM vs HAND classifier (LOSO) ...")
    feat_summary, feat_perfold = pain_prob_as_feature_loso(probs)
    feat_perfold.to_csv(TAB_DIR / "tierA3_stage1_as_feature_loso_perfold.csv",
                        index=False)
    feat_path = TAB_DIR / "tierA3_stage1_as_feature_loso.csv"
    pd.DataFrame([feat_summary]).to_csv(feat_path, index=False)
    print(f"[save] {feat_path}")

    # Plots
    print("[plot] violin / per-subject diff / KDE / calibration ...")
    plot_violin_by_class(probs)
    plot_per_subject_diff(probs)
    plot_kde_arm_vs_hand(probs)
    plot_calibration(probs)

    # Report
    report_md = build_report(drift_train, drift_val, corr_df, feat_summary)
    report_path = REPORT_DIR / "15_tierA3_stage1_calibration_summary.md"
    report_path.write_text(report_md)
    print(f"[save] {report_path}")

    # Console summary
    print("\n=== SUMMARY ===")
    print(f"Train Wilcoxon p (ARM vs HAND mean pain_prob): "
          f"{drift_train['wilcoxon_p']:.3g}  | "
          f"Cliff's δ = {drift_train['cliffs_delta_arm_vs_hand']:+.3f}")
    pb = corr_df.loc[(corr_df['scope'] == 'global')
                     & (corr_df['split'] == 'train'),
                     'pointbiserial_r'].iloc[0]
    print(f"Global point-biserial r (train, pain segments): {pb:+.4f}")
    print(f"pain_prob-alone macro-F1: {feat_summary['macro_f1_mean']:.3f} "
          f"± {feat_summary['macro_f1_std']:.3f}")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
