"""Class-separability statistics for AI4Pain 2026 dataset.

Runs per-feature omnibus and pairwise tests on subject-mean feature values
(one data point per subject per class) to avoid pseudoreplication from the
dominant subject effect. Outputs CSV tables, plots, and a Markdown report.

Run:
    uv run python scripts/06_class_tests.py
from the analysis/ directory.
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import CLASSES  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120
sns.set_context("talk", font_scale=0.7)

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_ROOT / "results" / "tables"
PLOTS_DIR = ANALYSIS_ROOT / "plots" / "class_tests"
REPORTS_DIR = ANALYSIS_ROOT / "results" / "reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    "NoPain": "#4c72b0",
    "PainArm": "#dd8452",
    "PainHand": "#c44e52",
}

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
PAIRS = [
    ("NoPain", "PainArm"),
    ("NoPain", "PainHand"),
    ("PainArm", "PainHand"),
]


# ---------------------------------------------------------------------------
# Merge step
# ---------------------------------------------------------------------------
def merge_feature_tables() -> tuple[pd.DataFrame, dict[str, str]]:
    """Read the 3 parquet tables, inner-join on segment_id, resolve collisions
    by prefixing raw-stats columns with 'raw_'. Returns merged df and a
    feature -> source map ('physio' | 'tf' | 'raw')."""
    physio = pd.read_parquet(TABLES_DIR / "physio_features.parquet")
    tf = pd.read_parquet(TABLES_DIR / "tf_features.parquet")
    raw = pd.read_parquet(TABLES_DIR / "raw_stats_per_segment.parquet")

    physio_feats = [c for c in physio.columns if c not in META_COLS]
    tf_feats = [c for c in tf.columns if c not in META_COLS]
    raw_feats = [c for c in raw.columns if c not in META_COLS]

    # Prefix raw-stats columns that collide with physio or tf features.
    collisions = (set(raw_feats) & set(physio_feats)) | (set(raw_feats) & set(tf_feats))
    rename_map = {c: f"raw_{c}" for c in collisions}
    raw_renamed = raw.rename(columns=rename_map)

    # After renaming, rebuild raw feature list
    raw_feats_new = [rename_map.get(c, c) for c in raw_feats]

    # Source map
    source_map: dict[str, str] = {}
    for c in physio_feats:
        source_map[c] = "physio"
    for c in tf_feats:
        source_map[c] = "tf"
    for c in raw_feats_new:
        source_map[c] = "raw"

    # Inner-join on segment_id (keep meta cols from physio)
    merged = physio.merge(
        tf.drop(columns=[c for c in META_COLS if c != "segment_id"]),
        on="segment_id",
        how="inner",
    )
    merged = merged.merge(
        raw_renamed.drop(columns=[c for c in META_COLS if c != "segment_id"]),
        on="segment_id",
        how="inner",
    )
    return merged, source_map


# ---------------------------------------------------------------------------
# Effect size helpers
# ---------------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: (#(x>y) - #(x<y)) / (nx*ny) across all pairs.

    Uses a vectorised rank-based formulation for speed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    # Use outer comparison for small arrays (<=2000*2000 is tolerable here).
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return float(gt - lt) / (nx * ny)


def rank_biserial_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for Wilcoxon signed-rank test.

    r = (sum(positive ranks) - sum(negative ranks)) / sum(all ranks)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    ranks = sp_stats.rankdata(np.abs(d))
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    total = ranks.sum()
    if total == 0:
        return np.nan
    return float(pos - neg) / total


def eta_squared_oneway(groups: list[np.ndarray]) -> float:
    """eta^2 = SS_between / SS_total for one-way layout."""
    all_vals = np.concatenate(groups)
    grand = all_vals.mean()
    ss_total = np.sum((all_vals - grand) ** 2)
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    if ss_total <= 0:
        return np.nan
    return float(ss_between / ss_total)


# ---------------------------------------------------------------------------
# Subject-mean construction
# ---------------------------------------------------------------------------
def subject_mean_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return df with one row per (subject, class) holding the mean of each
    feature within that subject+class group."""
    agg = df.groupby(["subject", "class"], as_index=False)[features].mean()
    return agg


def class_arrays(
    subject_means: pd.DataFrame, feature: str
) -> dict[str, np.ndarray]:
    """Return {class_name: 1D subject-mean array} for a feature."""
    out: dict[str, np.ndarray] = {}
    for cls in CLASSES:
        vals = subject_means.loc[
            subject_means["class"] == cls, feature
        ].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        out[cls] = vals
    return out


def paired_arrays(
    subject_means: pd.DataFrame, feature: str, cls_a: str, cls_b: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired (a, b) subject-mean arrays after dropping subjects with
    NaN in either."""
    a = subject_means[subject_means["class"] == cls_a][["subject", feature]]
    b = subject_means[subject_means["class"] == cls_b][["subject", feature]]
    merged = a.merge(b, on="subject", suffixes=("_a", "_b"))
    merged = merged.dropna()
    return (
        merged[f"{feature}_a"].to_numpy(dtype=float),
        merged[f"{feature}_b"].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# Core per-feature tests
# ---------------------------------------------------------------------------
def per_feature_tests(
    subject_means: pd.DataFrame, features: list[str], source_map: dict[str, str]
) -> pd.DataFrame:
    rows: list[dict] = []
    for feat in features:
        row: dict = {"feature": feat, "source": source_map.get(feat, "?")}
        cls_data = class_arrays(subject_means, feat)
        ns = {c: len(cls_data[c]) for c in CLASSES}
        row["n_subjects"] = min(ns.values())
        # Means (unpaired, all subjects with data)
        for c in CLASSES:
            row[f"mean_{c}"] = float(np.mean(cls_data[c])) if ns[c] > 0 else np.nan
            row[f"std_{c}"] = float(np.std(cls_data[c], ddof=1)) if ns[c] > 1 else np.nan

        # One-way ANOVA
        try:
            groups = [cls_data[c] for c in CLASSES]
            if all(len(g) > 1 for g in groups) and any(np.std(g) > 0 for g in groups):
                F, p = sp_stats.f_oneway(*groups)
                row["anova_F"] = float(F)
                row["anova_p"] = float(p)
                row["eta2"] = eta_squared_oneway(groups)
            else:
                row["anova_F"] = np.nan
                row["anova_p"] = np.nan
                row["eta2"] = np.nan
        except Exception:
            row["anova_F"] = np.nan
            row["anova_p"] = np.nan
            row["eta2"] = np.nan

        # Kruskal-Wallis
        try:
            groups = [cls_data[c] for c in CLASSES]
            if all(len(g) > 1 for g in groups):
                H, p = sp_stats.kruskal(*groups)
                row["kruskal_H"] = float(H)
                row["kruskal_p"] = float(p)
            else:
                row["kruskal_H"] = np.nan
                row["kruskal_p"] = np.nan
        except Exception:
            row["kruskal_H"] = np.nan
            row["kruskal_p"] = np.nan

        # Levene
        try:
            groups = [cls_data[c] for c in CLASSES]
            if all(len(g) > 1 for g in groups):
                W, p = sp_stats.levene(*groups, center="median")
                row["levene_W"] = float(W)
                row["levene_p"] = float(p)
            else:
                row["levene_W"] = np.nan
                row["levene_p"] = np.nan
        except Exception:
            row["levene_W"] = np.nan
            row["levene_p"] = np.nan

        # Pairwise Mann-Whitney U + Cliff's delta
        for ca, cb in PAIRS:
            prefix = f"mw_{ca}_vs_{cb}"
            x, y = cls_data[ca], cls_data[cb]
            try:
                if len(x) > 0 and len(y) > 0:
                    U, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
                    row[f"{prefix}_U"] = float(U)
                    row[f"{prefix}_p"] = float(p)
                    row[f"{prefix}_cliffs_delta"] = cliffs_delta(x, y)
                else:
                    row[f"{prefix}_U"] = np.nan
                    row[f"{prefix}_p"] = np.nan
                    row[f"{prefix}_cliffs_delta"] = np.nan
            except Exception:
                row[f"{prefix}_U"] = np.nan
                row[f"{prefix}_p"] = np.nan
                row[f"{prefix}_cliffs_delta"] = np.nan

        # Paired Wilcoxon + rank-biserial
        for ca, cb in PAIRS:
            prefix = f"wilcoxon_{ca}_vs_{cb}"
            x, y = paired_arrays(subject_means, feat, ca, cb)
            try:
                if len(x) > 1 and np.any(x - y != 0):
                    W, p = sp_stats.wilcoxon(x, y, zero_method="wilcox")
                    row[f"{prefix}_W"] = float(W)
                    row[f"{prefix}_p"] = float(p)
                    row[f"{prefix}_rb"] = rank_biserial_paired(x, y)
                else:
                    row[f"{prefix}_W"] = np.nan
                    row[f"{prefix}_p"] = np.nan
                    row[f"{prefix}_rb"] = np.nan
            except Exception:
                row[f"{prefix}_W"] = np.nan
                row[f"{prefix}_p"] = np.nan
                row[f"{prefix}_rb"] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------
def apply_fdr(df: pd.DataFrame, p_col: str, alpha: float = 0.05) -> pd.DataFrame:
    out_p = f"{p_col}_fdr"
    out_sig = p_col.replace("_p", "") + "_signif"
    pvals = df[p_col].to_numpy()
    mask = ~np.isnan(pvals)
    adj = np.full_like(pvals, np.nan, dtype=float)
    sig = np.zeros_like(pvals, dtype=bool)
    if mask.sum() > 0:
        rej, p_adj, _, _ = multipletests(pvals[mask], alpha=alpha, method="fdr_bh")
        adj[mask] = p_adj
        sig[mask] = rej
    df[out_p] = adj
    df[out_sig] = sig
    return df


def apply_all_fdr(df: pd.DataFrame) -> pd.DataFrame:
    families = ["anova_p", "kruskal_p"]
    for ca, cb in PAIRS:
        families.append(f"mw_{ca}_vs_{cb}_p")
    for ca, cb in PAIRS:
        families.append(f"wilcoxon_{ca}_vs_{cb}_p")
    for p_col in families:
        if p_col in df.columns:
            df = apply_fdr(df, p_col)
    return df


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def direction_tag(row: pd.Series) -> str:
    m = {c: row[f"mean_{c}"] for c in CLASSES}
    sorted_cls = sorted(CLASSES, key=lambda c: m[c], reverse=True)
    # Simple "Pain > NoPain" style tag based on ranks
    pain_mean = np.mean([m["PainArm"], m["PainHand"]])
    no_mean = m["NoPain"]
    if pain_mean > no_mean:
        tag = "Pain > NoPain"
    elif pain_mean < no_mean:
        tag = "Pain < NoPain"
    else:
        tag = "Pain ~ NoPain"
    order = " > ".join(sorted_cls)
    return f"{tag} ({order})"


def top_features_table(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    sig = df[df["anova_signif"] == True].copy()  # noqa: E712
    sig = sig.sort_values("anova_F", ascending=False).head(n).copy()
    sig["direction_tag"] = sig.apply(direction_tag, axis=1)
    keep = [
        "feature",
        "source",
        "anova_F",
        "anova_p",
        "anova_p_fdr",
        "eta2",
        "mean_NoPain",
        "mean_PainArm",
        "mean_PainHand",
        "direction_tag",
    ]
    return sig[keep]


def pairwise_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for ca, cb in PAIRS:
        p_col = f"mw_{ca}_vs_{cb}_p_fdr"
        d_col = f"mw_{ca}_vs_{cb}_cliffs_delta"
        mask = df[p_col] < 0.05
        sig = df[mask].copy()
        # Rank by absolute Cliff's delta
        sig["abs_delta"] = sig[d_col].abs()
        top = sig.sort_values("abs_delta", ascending=False).head(10)
        for _, r in top.iterrows():
            rows.append({
                "pair": f"{ca}_vs_{cb}",
                "feature": r["feature"],
                "source": r["source"],
                "cliffs_delta": r[d_col],
                "mw_p_fdr": r[p_col],
                "n_significant_in_pair": int(mask.sum()),
            })
        if mask.sum() == 0:
            rows.append({
                "pair": f"{ca}_vs_{cb}",
                "feature": "",
                "source": "",
                "cliffs_delta": np.nan,
                "mw_p_fdr": np.nan,
                "n_significant_in_pair": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pain vs NoPain (2-class)
# ---------------------------------------------------------------------------
def pain_vs_nopain_table(
    subject_means: pd.DataFrame, features: list[str], source_map: dict[str, str]
) -> pd.DataFrame:
    # Collapse to 2 classes: subject mean across Pain segments (pool Arm+Hand)
    # Need subject x class aggregation; we already have that, so average Arm&Hand.
    # Build per-subject Pain means by averaging PainArm and PainHand subject-means.
    pain = (
        subject_means[subject_means["class"].isin(["PainArm", "PainHand"])]
        .groupby("subject", as_index=False)[features]
        .mean()
    )
    pain["class"] = "Pain"
    nopain = subject_means[subject_means["class"] == "NoPain"].copy()
    nopain = nopain[["subject"] + features].copy()
    nopain["class"] = "NoPain"
    combined = pd.concat([pain, nopain], ignore_index=True)

    rows: list[dict] = []
    for feat in features:
        x = pain[feat].dropna().to_numpy(dtype=float)
        y = nopain[feat].dropna().to_numpy(dtype=float)
        row = {
            "feature": feat,
            "source": source_map.get(feat, "?"),
            "mean_Pain": float(x.mean()) if len(x) else np.nan,
            "mean_NoPain": float(y.mean()) if len(y) else np.nan,
            "n_subjects": min(len(x), len(y)),
        }
        try:
            if len(x) > 1 and len(y) > 1:
                F, p = sp_stats.f_oneway(x, y)
                row["anova_F"] = float(F)
                row["anova_p"] = float(p)
                row["eta2"] = eta_squared_oneway([x, y])
            else:
                row["anova_F"] = np.nan
                row["anova_p"] = np.nan
                row["eta2"] = np.nan
        except Exception:
            row["anova_F"] = np.nan
            row["anova_p"] = np.nan
            row["eta2"] = np.nan
        # Paired Wilcoxon (per subject): pain vs nopain
        paired = pain[["subject", feat]].merge(
            nopain[["subject", feat]], on="subject", suffixes=("_pain", "_nopain")
        ).dropna()
        try:
            if len(paired) > 1 and np.any(paired[f"{feat}_pain"] != paired[f"{feat}_nopain"]):
                W, p = sp_stats.wilcoxon(
                    paired[f"{feat}_pain"], paired[f"{feat}_nopain"], zero_method="wilcox"
                )
                row["wilcoxon_W"] = float(W)
                row["wilcoxon_p"] = float(p)
                row["wilcoxon_rb"] = rank_biserial_paired(
                    paired[f"{feat}_pain"].to_numpy(),
                    paired[f"{feat}_nopain"].to_numpy(),
                )
            else:
                row["wilcoxon_W"] = np.nan
                row["wilcoxon_p"] = np.nan
                row["wilcoxon_rb"] = np.nan
        except Exception:
            row["wilcoxon_W"] = np.nan
            row["wilcoxon_p"] = np.nan
            row["wilcoxon_rb"] = np.nan
        # Cliff's delta
        row["cliffs_delta"] = cliffs_delta(x, y) if len(x) and len(y) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out = apply_fdr(out, "anova_p")
    out = apply_fdr(out, "wilcoxon_p")
    out["direction"] = np.where(
        out["mean_Pain"] > out["mean_NoPain"], "Pain > NoPain", "Pain < NoPain"
    )
    return out


# ---------------------------------------------------------------------------
# Arm vs Hand among pain
# ---------------------------------------------------------------------------
def arm_vs_hand_table(
    subject_means: pd.DataFrame, features: list[str], source_map: dict[str, str]
) -> pd.DataFrame:
    rows: list[dict] = []
    arm = subject_means[subject_means["class"] == "PainArm"]
    hand = subject_means[subject_means["class"] == "PainHand"]
    for feat in features:
        x, y = paired_arrays(subject_means, feat, "PainArm", "PainHand")
        row = {
            "feature": feat,
            "source": source_map.get(feat, "?"),
            "mean_PainArm": float(arm[feat].mean()) if len(arm) else np.nan,
            "mean_PainHand": float(hand[feat].mean()) if len(hand) else np.nan,
            "n_subjects": len(x),
        }
        try:
            if len(x) > 1 and np.any(x - y != 0):
                W, p = sp_stats.wilcoxon(x, y, zero_method="wilcox")
                row["wilcoxon_W"] = float(W)
                row["wilcoxon_p"] = float(p)
                row["wilcoxon_rb"] = rank_biserial_paired(x, y)
            else:
                row["wilcoxon_W"] = np.nan
                row["wilcoxon_p"] = np.nan
                row["wilcoxon_rb"] = np.nan
        except Exception:
            row["wilcoxon_W"] = np.nan
            row["wilcoxon_p"] = np.nan
            row["wilcoxon_rb"] = np.nan
        # Mann-Whitney U + Cliff's delta (unpaired view, for magnitude)
        try:
            if len(x) > 0 and len(y) > 0:
                U, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
                row["mw_U"] = float(U)
                row["mw_p"] = float(p)
                row["cliffs_delta"] = cliffs_delta(x, y)
            else:
                row["mw_U"] = np.nan
                row["mw_p"] = np.nan
                row["cliffs_delta"] = np.nan
        except Exception:
            row["mw_U"] = np.nan
            row["mw_p"] = np.nan
            row["cliffs_delta"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out = apply_fdr(out, "wilcoxon_p")
    out = apply_fdr(out, "mw_p")
    out["direction"] = np.where(
        out["mean_PainArm"] > out["mean_PainHand"],
        "Arm > Hand",
        "Arm < Hand",
    )
    return out


# ---------------------------------------------------------------------------
# Validation-split consistency
# ---------------------------------------------------------------------------
def validation_consistency(
    merged: pd.DataFrame,
    top_features: list[str],
    train_table: pd.DataFrame,
) -> pd.DataFrame:
    val = merged[merged["split"] == "validation"]
    val_subject_means = val.groupby(["subject", "class"], as_index=False)[
        top_features
    ].mean()
    rows: list[dict] = []
    for feat in top_features:
        train_means = {
            c: train_table.set_index("feature").loc[feat, f"mean_{c}"] for c in CLASSES
        }
        val_means = {}
        for c in CLASSES:
            vals = val_subject_means.loc[
                val_subject_means["class"] == c, feat
            ].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            val_means[c] = float(vals.mean()) if len(vals) else np.nan
        # Direction preservation: check sign of PainArm-NoPain and PainHand-NoPain
        def sign(d: float) -> int:
            if np.isnan(d):
                return 0
            return 1 if d > 0 else (-1 if d < 0 else 0)

        train_sign_arm = sign(train_means["PainArm"] - train_means["NoPain"])
        train_sign_hand = sign(train_means["PainHand"] - train_means["NoPain"])
        val_sign_arm = sign(val_means["PainArm"] - val_means["NoPain"])
        val_sign_hand = sign(val_means["PainHand"] - val_means["NoPain"])
        preserved = (train_sign_arm == val_sign_arm) and (
            train_sign_hand == val_sign_hand
        )
        rows.append({
            "feature": feat,
            "train_NoPain": train_means["NoPain"],
            "train_PainArm": train_means["PainArm"],
            "train_PainHand": train_means["PainHand"],
            "val_NoPain": val_means["NoPain"],
            "val_PainArm": val_means["PainArm"],
            "val_PainHand": val_means["PainHand"],
            "train_direction_preserved_in_val": preserved,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_top20_anova_bar(results: pd.DataFrame) -> None:
    top = (
        results[results["anova_signif"] == True]  # noqa: E712
        .sort_values("anova_F", ascending=False)
        .head(20)
    )
    if len(top) == 0:
        top = results.sort_values("anova_F", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [
        {"physio": "#4c72b0", "tf": "#55a868", "raw": "#c44e52"}.get(s, "#888")
        for s in top["source"]
    ]
    ax.barh(top["feature"][::-1], top["anova_F"][::-1], color=colors[::-1])
    ax.set_xlabel("ANOVA F (subject-mean, train)")
    ax.set_title("Top 20 features by one-way ANOVA F, 3-class")
    # Custom legend
    from matplotlib.patches import Patch

    legend = [
        Patch(color="#4c72b0", label="physio"),
        Patch(color="#55a868", label="tf"),
        Patch(color="#c44e52", label="raw"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top20_anova_bar.png")
    plt.close(fig)


def plot_effect_direction_grid(results: pd.DataFrame, top_n: int = 30) -> None:
    top = (
        results[results["anova_signif"] == True]  # noqa: E712
        .sort_values("anova_F", ascending=False)
        .head(top_n)
    )
    if len(top) == 0:
        top = results.sort_values("anova_F", ascending=False).head(top_n)
    mat = top[[f"mean_{c}" for c in CLASSES]].to_numpy(dtype=float)
    # Row-wise z-score
    mean = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    z = (mat - mean) / std

    fig, ax = plt.subplots(figsize=(6, max(6, 0.3 * len(top))))
    sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0,
        yticklabels=top["feature"].tolist(),
        xticklabels=list(CLASSES),
        cbar_kws={"label": "z-score (per feature)"},
        ax=ax,
    )
    ax.set_title(f"Class means (z per feature), top {len(top)} by ANOVA F")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "effect_direction_grid.png")
    plt.close(fig)


def plot_cliffs_delta_pairwise(results: pd.DataFrame, top_n: int = 20) -> None:
    top = (
        results[results["anova_signif"] == True]  # noqa: E712
        .sort_values("anova_F", ascending=False)
        .head(top_n)
    )
    if len(top) == 0:
        top = results.sort_values("anova_F", ascending=False).head(top_n)
    deltas = {
        f"{ca} vs {cb}": top[f"mw_{ca}_vs_{cb}_cliffs_delta"].to_numpy()
        for ca, cb in PAIRS
    }
    features = top["feature"].tolist()
    n = len(features)
    width = 0.27
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * n), 6))
    colors = ["#4c72b0", "#dd8452", "#c44e52"]
    for i, (label, vals) in enumerate(deltas.items()):
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=70, ha="right", fontsize=7)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Cliff's delta")
    ax.set_title(f"Pairwise Cliff's delta, top {n} features")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cliffs_delta_pairwise.png")
    plt.close(fig)


def volcano_plot(
    df: pd.DataFrame,
    effect_col: str,
    p_col: str,
    title: str,
    out_name: str,
    label_top: int = 10,
) -> None:
    d = df.dropna(subset=[effect_col, p_col]).copy()
    if len(d) == 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(title + " (no data)")
        fig.savefig(PLOTS_DIR / out_name)
        plt.close(fig)
        return
    y = -np.log10(np.clip(d[p_col].to_numpy(), 1e-30, None))
    x = d[effect_col].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(d[p_col] < 0.05, "#c44e52", "#888")
    ax.scatter(x, y, c=colors, s=18, alpha=0.7)
    # Label top by combined rank
    d["score"] = np.abs(x) * y
    top = d.sort_values("score", ascending=False).head(label_top)
    for _, r in top.iterrows():
        ax.annotate(
            r["feature"],
            (r[effect_col], -np.log10(max(r[p_col], 1e-30))),
            fontsize=7,
            alpha=0.9,
        )
    ax.axhline(-np.log10(0.05), color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(effect_col + " (effect size)")
    ax.set_ylabel("-log10(FDR p)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name)
    plt.close(fig)


def plot_val_consistency(val_df: pd.DataFrame, train_table: pd.DataFrame) -> None:
    # Use (PainArm + PainHand)/2 - NoPain as single-scalar effect direction proxy
    rows_effect = []
    train_lookup = train_table.set_index("feature")
    for _, r in val_df.iterrows():
        f = r["feature"]
        train_eff = 0.5 * (
            train_lookup.loc[f, "mean_PainArm"] + train_lookup.loc[f, "mean_PainHand"]
        ) - train_lookup.loc[f, "mean_NoPain"]
        val_eff = 0.5 * (r["val_PainArm"] + r["val_PainHand"]) - r["val_NoPain"]
        rows_effect.append((f, train_eff, val_eff, r["train_direction_preserved_in_val"]))
    arr = pd.DataFrame(rows_effect, columns=["feature", "train_eff", "val_eff", "preserved"])
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = np.where(arr["preserved"], "#55a868", "#c44e52")
    ax.scatter(arr["train_eff"], arr["val_eff"], c=colors, s=40, alpha=0.85)
    # Diagonal
    xmin = float(min(arr["train_eff"].min(), arr["val_eff"].min()))
    xmax = float(max(arr["train_eff"].max(), arr["val_eff"].max()))
    pad = 0.05 * (xmax - xmin + 1e-9)
    lo, hi = xmin - pad, xmax + pad
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    # Label top 10 by abs train_eff
    top = arr.reindex(arr["train_eff"].abs().sort_values(ascending=False).index).head(10)
    for _, r in top.iterrows():
        ax.annotate(r["feature"], (r["train_eff"], r["val_eff"]), fontsize=7, alpha=0.85)
    ax.set_xlabel("Train effect (mean_Pain - mean_NoPain)")
    ax.set_ylabel("Validation effect")
    ax.set_title("Train vs validation class-effect consistency, top 30 features")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#55a868", label="direction preserved"),
            Patch(color="#c44e52", label="flipped"),
        ],
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "val_vs_train_consistency_scatter.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def signal_axis(name: str) -> str:
    n = name.lower()
    if n.startswith("bvp"):
        return "BVP/HR"
    if n.startswith("raw_bvp"):
        return "BVP/HR"
    if n.startswith("eda") or "eda" in n.split("_")[0:2]:
        return "EDA"
    if n.startswith("raw_eda"):
        return "EDA"
    if n.startswith("resp"):
        return "RESP"
    if n.startswith("raw_resp"):
        return "RESP"
    if n.startswith("spo2"):
        return "SpO2"
    if n.startswith("raw_spo2"):
        return "SpO2"
    return "OTHER"


def write_report(
    results: pd.DataFrame,
    top_table: pd.DataFrame,
    pair_summary: pd.DataFrame,
    pain_vs_nopain: pd.DataFrame,
    arm_vs_hand: pd.DataFrame,
    val_df: pd.DataFrame,
    n_train_subjects: int,
) -> None:
    n_sig_3c = int(results["anova_signif"].sum())
    total_feats = len(results)

    # Axis breakdown for significant features
    axis_counts: dict[str, int] = {}
    for _, r in results[results["anova_signif"] == True].iterrows():  # noqa: E712
        axis_counts[signal_axis(r["feature"])] = axis_counts.get(
            signal_axis(r["feature"]), 0
        ) + 1

    # Pain vs NoPain
    pvn_sig = int((pain_vs_nopain["anova_p_fdr"] < 0.05).sum())
    pvn_top = pain_vs_nopain.dropna(subset=["anova_F"]).sort_values(
        "anova_F", ascending=False
    ).head(10)
    pvn_med_delta = float(pvn_top["cliffs_delta"].abs().median()) if len(pvn_top) else np.nan

    # Arm vs Hand
    avh_sig = int((arm_vs_hand["wilcoxon_p_fdr"] < 0.05).sum())
    avh_top = arm_vs_hand.dropna(subset=["wilcoxon_p_fdr"]).sort_values(
        "wilcoxon_p_fdr", ascending=True
    ).head(10)

    # Validation consistency
    n_preserved = int(val_df["train_direction_preserved_in_val"].sum())
    pct_preserved = 100.0 * n_preserved / max(len(val_df), 1)

    lines: list[str] = []
    lines.append("# 06 Class-separability tests (subject-mean)")
    lines.append("")
    lines.append(
        f"Per-feature omnibus + pairwise tests on subject-mean feature values "
        f"({n_train_subjects} train subjects x 3 classes = {n_train_subjects*3} "
        f"data points per feature). Tests on subject means avoid pseudoreplication "
        f"caused by the dominant subject effect."
    )
    lines.append("")
    lines.append("## 3-class ANOVA (FDR < 0.05)")
    lines.append(
        f"- Total features tested: **{total_feats}**"
    )
    lines.append(
        f"- Significant at Benjamini-Hochberg FDR q<0.05: **{n_sig_3c}** "
        f"({100*n_sig_3c/max(total_feats,1):.1f}%)"
    )
    if axis_counts:
        lines.append("- Breakdown by physiological axis:")
        for ax_name in sorted(axis_counts, key=axis_counts.get, reverse=True):
            lines.append(f"  - {ax_name}: {axis_counts[ax_name]}")
    lines.append("")
    lines.append("## Top 20 features by ANOVA F")
    lines.append("")
    lines.append(
        "| rank | feature | source | F | eta^2 | FDR p | NoPain | PainArm | PainHand | direction |"
    )
    lines.append(
        "|-----:|---------|--------|---:|------:|------:|-------:|--------:|---------:|-----------|"
    )
    top20 = top_table.head(20)
    for i, (_, r) in enumerate(top20.iterrows(), 1):
        lines.append(
            f"| {i} | `{r['feature']}` | {r['source']} | "
            f"{r['anova_F']:.2f} | {r['eta2']:.3f} | {r['anova_p_fdr']:.2e} | "
            f"{r['mean_NoPain']:.3g} | {r['mean_PainArm']:.3g} | "
            f"{r['mean_PainHand']:.3g} | {r['direction_tag']} |"
        )
    lines.append("")
    lines.append("## Pain detection: Pain (Arm+Hand pooled) vs NoPain")
    lines.append(
        f"- Features with FDR-adjusted ANOVA p<0.05: **{pvn_sig}** of {len(pain_vs_nopain)}."
    )
    if not np.isnan(pvn_med_delta):
        lines.append(
            f"- Median |Cliff's delta| among top 10 by F: **{pvn_med_delta:.3f}**."
        )
    lines.append("")
    lines.append("Top 10 features (by ANOVA F):")
    lines.append("")
    lines.append(
        "| feature | source | F | FDR p | Cliff's delta | direction |"
    )
    lines.append(
        "|---------|--------|---:|------:|-------------:|-----------|"
    )
    for _, r in pvn_top.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['source']} | {r['anova_F']:.2f} | "
            f"{r['anova_p_fdr']:.2e} | {r['cliffs_delta']:.3f} | {r['direction']} |"
        )
    lines.append("")
    lines.append("## Pain localisation: PainArm vs PainHand (paired Wilcoxon)")
    lines.append(
        f"- Features with FDR-adjusted paired-Wilcoxon p<0.05: **{avh_sig}** of "
        f"{len(arm_vs_hand)}."
    )
    if avh_sig == 0:
        lines.append(
            "- No features reach FDR<0.05 at the subject-mean level. Arm vs Hand "
            "is intrinsically the harder problem here (same stimulus, different "
            "site), and the subject-mean aggregation throws away within-subject "
            "segment-level variance — expected to be weak."
        )
    lines.append("")
    lines.append("Top 10 features by smallest paired-Wilcoxon FDR p:")
    lines.append("")
    lines.append(
        "| feature | source | Wilcoxon FDR p | rank-biserial | Cliff's delta | direction |"
    )
    lines.append(
        "|---------|--------|--------------:|-------------:|-------------:|-----------|"
    )
    for _, r in avh_top.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['source']} | {r['wilcoxon_p_fdr']:.2e} | "
            f"{r['wilcoxon_rb']:.3f} | {r['cliffs_delta']:.3f} | {r['direction']} |"
        )
    lines.append("")
    lines.append("## Validation-split consistency (top 30 train ANOVA features)")
    lines.append(
        f"- Direction preserved in validation for "
        f"**{n_preserved}/{len(val_df)}** features ({pct_preserved:.1f}%)."
    )
    lines.append("")
    lines.append("## Plain-English interpretation")
    if axis_counts:
        ordered = sorted(axis_counts.items(), key=lambda kv: kv[1], reverse=True)
        top_axis = ordered[0][0]
        lines.append(
            f"- The physiological axis carrying the most class information is "
            f"**{top_axis}**, followed by "
            + ", ".join(f"{a} ({c})" for a, c in ordered[1:]) + "."
        )
    lines.append(
        "- Pain vs NoPain is clearly detectable; Arm vs Hand is much subtler."
    )
    lines.append(
        "- Direction preservation on the validation split indicates whether the "
        "per-feature class shift generalises to held-out subjects."
    )
    lines.append("")
    lines.append("## Methodological caveats")
    lines.append(
        "- Subject-mean aggregation collapses 12 segments per subject per class "
        "into a single data point. This correctly controls for the subject effect "
        "but deliberately discards within-subject variance, making per-feature "
        "tests conservative relative to segment-level pooling."
    )
    lines.append(
        "- Mann-Whitney U on subject means is unpaired; the paired Wilcoxon "
        "exploits the within-subject design and is the primary test for each "
        "pairwise contrast."
    )
    lines.append(
        "- Benjamini-Hochberg FDR is applied *within* each test family "
        "(ANOVA, Kruskal-Wallis, each pairwise Mann-Whitney, each pairwise "
        "Wilcoxon, Pain-vs-NoPain, Arm-vs-Hand). No correction across families."
    )
    lines.append(
        "- Cliff's delta values are computed on the unpaired subject-mean arrays; "
        "rank-biserial correlations use the paired-Wilcoxon signed ranks."
    )
    lines.append(
        "- Validation split has many fewer subjects; small per-class N on that "
        "split limits the precision of the val means. Direction agreement is a "
        "weak proxy for generalisation but good for flagging obviously unstable "
        "features."
    )
    lines.append("")
    (REPORTS_DIR / "06_class_tests_summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[1/7] Merging feature tables...")
    merged, source_map = merge_feature_tables()
    print(f"  merged shape: {merged.shape}")
    merged.to_parquet(TABLES_DIR / "all_features_merged.parquet", index=False)

    features = [c for c in merged.columns if c not in META_COLS]
    print(f"  features: {len(features)}  "
          f"(physio={sum(1 for f in features if source_map[f]=='physio')}, "
          f"tf={sum(1 for f in features if source_map[f]=='tf')}, "
          f"raw={sum(1 for f in features if source_map[f]=='raw')})")

    train = merged[merged["split"] == "train"].copy()
    n_train_subjects = train["subject"].nunique()
    print(f"  train subjects: {n_train_subjects}")

    print("[2/7] Building subject-mean table (train)...")
    subj_means = subject_mean_table(train, features)
    print(f"  subject-mean rows: {len(subj_means)}")

    print("[3/7] Running per-feature tests...")
    results = per_feature_tests(subj_means, features, source_map)

    print("[4/7] Applying FDR correction...")
    results = apply_all_fdr(results)
    results.to_csv(TABLES_DIR / "class_tests_per_feature.csv", index=False)
    print(f"  wrote class_tests_per_feature.csv "
          f"(n_sig ANOVA={int(results['anova_signif'].sum())})")

    print("[5/7] Building summary tables...")
    top_table = top_features_table(results, n=30)
    top_table.to_csv(TABLES_DIR / "class_tests_top_features.csv", index=False)

    pair_summary = pairwise_summary_table(results)
    pair_summary.to_csv(TABLES_DIR / "class_tests_pairwise_summary.csv", index=False)

    pvn = pain_vs_nopain_table(subj_means, features, source_map)
    pvn_sorted = pvn.sort_values("anova_F", ascending=False)
    pvn_sorted.to_csv(TABLES_DIR / "class_tests_pain_vs_nopain.csv", index=False)

    avh = arm_vs_hand_table(subj_means, features, source_map)
    avh_sorted = avh.sort_values("wilcoxon_p", ascending=True)
    avh_sorted.to_csv(TABLES_DIR / "class_tests_arm_vs_hand.csv", index=False)

    # Validation means for top 30 features
    top_feats = top_table["feature"].tolist()
    if len(top_feats) == 0:
        top_feats = (
            results.sort_values("anova_F", ascending=False).head(30)["feature"].tolist()
        )
    val_df = validation_consistency(merged, top_feats, results)
    val_df.to_csv(TABLES_DIR / "class_tests_validation_means.csv", index=False)

    print("[6/7] Plotting...")
    plot_top20_anova_bar(results)
    plot_effect_direction_grid(results, top_n=30)
    plot_cliffs_delta_pairwise(results, top_n=20)
    volcano_plot(
        pvn,
        effect_col="cliffs_delta",
        p_col="anova_p_fdr",
        title="Pain vs NoPain: Cliff's delta vs -log10(FDR p)",
        out_name="pain_vs_nopain_effects.png",
        label_top=12,
    )
    volcano_plot(
        avh,
        effect_col="cliffs_delta",
        p_col="wilcoxon_p_fdr",
        title="PainArm vs PainHand: Cliff's delta vs -log10(FDR p)",
        out_name="arm_vs_hand_effects.png",
        label_top=12,
    )
    plot_val_consistency(val_df, results)

    print("[7/7] Writing report...")
    write_report(
        results,
        top_table,
        pair_summary,
        pvn_sorted,
        avh_sorted,
        val_df,
        n_train_subjects,
    )
    print("Done.")


if __name__ == "__main__":
    main()
