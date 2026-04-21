"""Subject diagnostics: flag non-responders, sign-flippers, low-signal subjects.

Outputs per-subject:
- class_separability: Cohen's d between NoPain and Pain (within-subject, averaged over features)
- sign_consistency: fraction of features where subject's direction matches group consensus
- within_subject_variance: mean feature variance (low = weak response)
- signal_quality: raw signal energy / artifact proxy
- LOSO F1 (if available from prior run)
- non_responder_flag: combined diagnostic
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.final_pipeline import (
    ARM_HAND, channel_of, load_clean_features, unique_by_canonical,
)

OUT = Path("results/final/subject_diagnostics")
OUT.mkdir(parents=True, exist_ok=True)


def per_subject_separability(df, feat_cols, pos="Pain"):
    """Cohen's d per-feature per-subject, aggregated."""
    rows = []
    for subj in sorted(df["subject"].unique()):
        g = df[df["subject"] == subj]
        if pos == "Pain":
            a = g[g["class"] == "NoPain"][feat_cols]
            b = g[g["class"] != "NoPain"][feat_cols]
        else:
            a = g[g["class"] == "PainArm"][feat_cols]
            b = g[g["class"] == "PainHand"][feat_cols]
        if len(a) < 2 or len(b) < 2:
            continue
        diff = (b.mean() - a.mean()).to_numpy()
        pooled = np.sqrt((a.var() + b.var()) / 2).to_numpy()
        pooled = np.where(pooled > 1e-8, pooled, 1.0)
        d = diff / pooled
        rows.append({
            "subject": int(subj),
            "separability_mean": float(np.nanmean(np.abs(d))),
            "separability_median": float(np.nanmedian(np.abs(d))),
            "separability_top10pct": float(np.nanmean(np.sort(np.abs(d))[-max(1, len(d) // 10):])),
        })
    return pd.DataFrame(rows)


def sign_consistency(df, feat_cols):
    """For each subject: fraction of features where Cohen's d direction matches group consensus."""
    # Group consensus: median direction per feature
    group_d = []
    for subj in df["subject"].unique():
        g = df[df["subject"] == subj]
        a = g[g["class"] == "NoPain"][feat_cols]
        b = g[g["class"] != "NoPain"][feat_cols]
        if len(a) < 2 or len(b) < 2:
            continue
        diff = (b.mean() - a.mean()).to_numpy()
        pooled = np.sqrt((a.var() + b.var()) / 2).to_numpy()
        pooled = np.where(pooled > 1e-8, pooled, 1.0)
        d = diff / pooled
        group_d.append(d)
    group_d = np.array(group_d)
    consensus_sign = np.sign(np.nanmedian(group_d, axis=0))

    rows = []
    for subj in sorted(df["subject"].unique()):
        g = df[df["subject"] == subj]
        a = g[g["class"] == "NoPain"][feat_cols]
        b = g[g["class"] != "NoPain"][feat_cols]
        if len(a) < 2 or len(b) < 2:
            continue
        diff = (b.mean() - a.mean()).to_numpy()
        pooled = np.sqrt((a.var() + b.var()) / 2).to_numpy()
        pooled = np.where(pooled > 1e-8, pooled, 1.0)
        d = diff / pooled
        subj_sign = np.sign(d)
        match = (subj_sign == consensus_sign) & (subj_sign != 0)
        rows.append({
            "subject": int(subj),
            "sign_consistency": float(np.nanmean(match.astype(float))),
            "flipped_fraction": float(np.nanmean((subj_sign == -consensus_sign).astype(float))),
        })
    return pd.DataFrame(rows)


def within_subject_stats(df, feat_cols):
    rows = []
    for subj in sorted(df["subject"].unique()):
        g = df[df["subject"] == subj]
        X = g[feat_cols].to_numpy(dtype=np.float32)
        X = np.nan_to_num(X)
        cv = X.std(axis=0) / (np.abs(X.mean(axis=0)) + 1e-6)
        rows.append({
            "subject": int(subj),
            "n_windows": len(g),
            "feat_mean_std": float(X.std(axis=0).mean()),
            "feat_median_cv": float(np.nanmedian(cv)),
            "feat_outlier_frac": float(((np.abs(X - np.nanmedian(X, axis=0)) / (X.std(axis=0) + 1e-6)) > 3).mean()),
        })
    return pd.DataFrame(rows)


def signal_quality_per_modality(df, feat_cols):
    """For each subject + modality: mean std across windows of key amplitude features."""
    rows = []
    mod_features = {
        "bvp": [c for c in feat_cols if channel_of(c) == "bvp" and "std" in c.lower()],
        "eda": [c for c in feat_cols if channel_of(c) == "eda" and "std" in c.lower()],
        "resp": [c for c in feat_cols if channel_of(c) == "resp" and "std" in c.lower()],
        "spo2": [c for c in feat_cols if channel_of(c) == "spo2" and "std" in c.lower()],
    }
    for subj in sorted(df["subject"].unique()):
        g = df[df["subject"] == subj]
        row = {"subject": int(subj)}
        for mod, cols in mod_features.items():
            if cols:
                row[f"{mod}_signal_std"] = float(g[cols].mean(axis=0).mean())
            else:
                row[f"{mod}_signal_std"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def load_loso_f1():
    p = Path("results/final/final_pipeline/loso_per_subject.csv")
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        return df[["subject", "f1"]].rename(columns={"f1": "loso_f1"})
    return None


def classify_subjects(merged):
    """Combine diagnostics into non-responder flag."""
    # Rank by separability (low = non-responder)
    merged["sep_rank"] = merged["separability_mean"].rank(pct=True)
    # Rank by sign consistency (low = sign-flipper)
    merged["sign_rank"] = merged["sign_consistency"].rank(pct=True)
    # Combined score
    merged["responder_score"] = 0.6 * merged["sep_rank"] + 0.4 * merged["sign_rank"]
    # Flags
    merged["flag_low_separability"] = merged["sep_rank"] < 0.2
    merged["flag_sign_flipper"] = merged["sign_consistency"] < 0.5
    merged["flag_non_responder"] = merged["responder_score"] < 0.25
    return merged


def main():
    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    val = df_all[df_all["split"] == "validation"].reset_index(drop=True)

    bvp = unique_by_canonical([c for c in feat_cols if channel_of(c) == "bvp"])
    eda = unique_by_canonical([c for c in feat_cols if channel_of(c) == "eda"])
    s1_pool = bvp + eda

    # Train diagnostics
    print(">>> Computing train diagnostics...")
    sep_train = per_subject_separability(train, s1_pool, "Pain")
    sign_train = sign_consistency(train, s1_pool)
    within_train = within_subject_stats(train, feat_cols)
    sig_train = signal_quality_per_modality(train, feat_cols)

    # Val diagnostics
    print(">>> Computing val diagnostics...")
    sep_val = per_subject_separability(val, s1_pool, "Pain")
    sign_val = sign_consistency(val, s1_pool)
    within_val = within_subject_stats(val, feat_cols)
    sig_val = signal_quality_per_modality(val, feat_cols)

    # Merge
    for df in [sep_train, sign_train, within_train, sig_train]:
        df["split"] = "train"
    for df in [sep_val, sign_val, within_val, sig_val]:
        df["split"] = "val"

    sep = pd.concat([sep_train, sep_val], ignore_index=True)
    sign = pd.concat([sign_train, sign_val], ignore_index=True)
    within = pd.concat([within_train, within_val], ignore_index=True)
    sig = pd.concat([sig_train, sig_val], ignore_index=True)

    merged = sep.merge(sign, on=["subject", "split"]).merge(within, on=["subject", "split"]).merge(sig, on=["subject", "split"])
    merged = classify_subjects(merged)

    # Join LOSO F1
    loso = load_loso_f1()
    if loso is not None:
        merged = merged.merge(loso, on="subject", how="left")

    merged = merged.sort_values(["split", "responder_score"])
    merged.to_csv(OUT / "subject_diagnostics.csv", index=False)

    # Report
    flagged = merged[merged["flag_non_responder"]].copy()
    print(f"\n=== FLAGGED NON-RESPONDERS ({len(flagged)}) ===")
    cols = ["subject", "split", "separability_mean", "sign_consistency", "responder_score"]
    if "loso_f1" in merged.columns:
        cols.append("loso_f1")
    print(flagged[cols].to_string(index=False))

    # Plots
    # 1. Separability distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    for split, color in [("train", "steelblue"), ("val", "coral")]:
        sub = merged[merged["split"] == split]
        ax.scatter(sub["separability_mean"], sub["sign_consistency"], label=split, alpha=0.7, s=80, c=color)
        for _, r in sub.iterrows():
            ax.annotate(int(r["subject"]), (r["separability_mean"], r["sign_consistency"]), fontsize=7)
    ax.axvline(merged["separability_mean"].quantile(0.2), color="red", linestyle=":", alpha=0.5, label="20th percentile")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="sign 50%")
    ax.set_xlabel("Class separability (Cohen's d)")
    ax.set_ylabel("Sign consistency with group")
    ax.set_title("Subject response diagnostics")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "responder_scatter.png", dpi=140)
    plt.close(fig)

    # 2. Per-subject LOSO F1 vs responder_score (if available)
    if "loso_f1" in merged.columns:
        train_sub = merged[(merged["split"] == "train") & merged["loso_f1"].notna()]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(train_sub["responder_score"], train_sub["loso_f1"], s=80, alpha=0.7)
        for _, r in train_sub.iterrows():
            ax.annotate(int(r["subject"]), (r["responder_score"], r["loso_f1"]), fontsize=7)
        corr = train_sub[["responder_score", "loso_f1"]].corr().iloc[0, 1]
        ax.set_xlabel("Responder score")
        ax.set_ylabel("LOSO F1")
        ax.set_title(f"LOSO F1 vs responder score (corr={corr:.3f})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "responder_vs_loso.png", dpi=140)
        plt.close(fig)

    # 3. Signal quality heatmap
    sig_mat = merged[["subject", "split", "bvp_signal_std", "eda_signal_std", "resp_signal_std", "spo2_signal_std"]].copy()
    sig_mat = sig_mat.sort_values(["split", "subject"])
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = sig_mat.set_index(["split", "subject"])[["bvp_signal_std", "eda_signal_std", "resp_signal_std", "spo2_signal_std"]]
    # Normalize per-column for viz
    pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-6)
    sns.heatmap(pivot_norm.T, cmap="viridis", ax=ax, cbar_kws={"label": "normalized signal std"})
    ax.set_title("Per-subject per-modality signal amplitude (normalized)")
    ax.set_xlabel("subject (train | val)")
    fig.tight_layout()
    fig.savefig(OUT / "signal_quality_heatmap.png", dpi=140)
    plt.close(fig)

    # Markdown
    lines = ["# Subject Diagnostics", ""]
    lines.append(f"Total subjects: train={len(merged[merged['split']=='train'])}, val={len(merged[merged['split']=='val'])}")
    lines.append(f"Non-responders flagged: {merged['flag_non_responder'].sum()}")
    lines.append(f"Sign-flippers flagged: {merged['flag_sign_flipper'].sum()}")
    lines.append("")
    lines.append("## Flagged Non-Responders")
    lines.append("")
    lines.append(flagged[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Full Rankings (Top 15 worst responder score)")
    lines.append("")
    worst = merged.head(15)
    lines.append(worst[cols].to_markdown(index=False))
    lines.append("")
    if "loso_f1" in merged.columns:
        tr = merged[(merged["split"] == "train") & merged["loso_f1"].notna()]
        corr_sep = tr[["separability_mean", "loso_f1"]].corr().iloc[0, 1]
        corr_sign = tr[["sign_consistency", "loso_f1"]].corr().iloc[0, 1]
        corr_score = tr[["responder_score", "loso_f1"]].corr().iloc[0, 1]
        lines.append("## Correlations with LOSO F1")
        lines.append(f"- separability_mean: {corr_sep:+.3f}")
        lines.append(f"- sign_consistency: {corr_sign:+.3f}")
        lines.append(f"- responder_score: {corr_score:+.3f}")
    (OUT / "report.md").write_text("\n".join(lines))
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
