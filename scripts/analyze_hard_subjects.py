"""Find what makes hard-LOSO subjects different: correlate LOSO F1 with signal stats."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.final_pipeline import channel_of, load_clean_features


def main() -> None:
    out_dir = Path("results/final/hard_subjects")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-subject LOSO scores (baseline already computed)
    per_subj = pd.read_csv("results/final/loso_spread/baseline_loso_per_subject.csv")
    per_subj.columns = [c.lower() for c in per_subj.columns]

    # Load features
    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    train = df_all[df_all["split"] == "train"].reset_index(drop=True)

    # Signal quality: per-subject NaN frac, std across windows, range
    sig_stats = []
    bvp = [c for c in feat_cols if channel_of(c) == "bvp"]
    eda = [c for c in feat_cols if channel_of(c) == "eda"]
    resp = [c for c in feat_cols if channel_of(c) == "resp"]
    spo2 = [c for c in feat_cols if channel_of(c) == "spo2"]
    for subj in train["subject"].unique():
        g = train[train["subject"] == subj]
        row = {
            "subject": int(subj),
            "n_windows": len(g),
            "bvp_std_mean": g[bvp].std().mean() if bvp else 0,
            "eda_std_mean": g[eda].std().mean() if eda else 0,
            "resp_std_mean": g[resp].std().mean() if resp else 0,
            "spo2_std_mean": g[spo2].std().mean() if spo2 else 0,
            "feat_median_cv": float((g[feat_cols].std() / (g[feat_cols].mean().abs() + 1e-6)).median()),
            "feat_outlier_frac": float(((g[feat_cols] - g[feat_cols].median()).abs() / (g[feat_cols].std() + 1e-6) > 3).mean().mean()),
        }
        # class separability: cohen d between NoPain vs Pain for top BVP/EDA features
        nop = g[g["class"] == "NoPain"][bvp + eda]
        pain = g[g["class"] != "NoPain"][bvp + eda]
        if len(nop) > 1 and len(pain) > 1:
            pooled_std = np.sqrt((nop.var() + pain.var()) / 2).replace(0, 1)
            d = (pain.mean() - nop.mean()).abs() / pooled_std
            row["class_separability_max"] = float(d.max())
            row["class_separability_mean"] = float(d.mean())
            row["class_separability_top10"] = float(d.nlargest(10).mean())
        sig_stats.append(row)
    sig_df = pd.DataFrame(sig_stats)

    # Merge with LOSO scores
    merged = per_subj.merge(sig_df, on="subject", how="inner")
    merged = merged.sort_values("f1")
    merged.to_csv(out_dir / "subject_stats.csv", index=False)

    # Correlations
    numeric_cols = [c for c in merged.columns if c not in ("subject",) and merged[c].dtype != object]
    corr = merged[numeric_cols].corr()["f1"].drop("f1").sort_values()
    print("=== Correlation with LOSO F1 ===")
    print(corr.to_string())

    # Bottom 10 vs top 10 subjects
    bot = merged.head(10)
    top = merged.tail(10)
    print("\n=== BOTTOM 10 LOSO subjects ===")
    print(bot[["subject", "f1", "bvp_std_mean", "eda_std_mean", "resp_std_mean", "feat_outlier_frac", "class_separability_mean", "class_separability_top10"]].to_string(index=False))
    print("\n=== TOP 10 LOSO subjects ===")
    print(top[["subject", "f1", "bvp_std_mean", "eda_std_mean", "resp_std_mean", "feat_outlier_frac", "class_separability_mean", "class_separability_top10"]].to_string(index=False))

    # Plot: class_separability vs F1
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, ["class_separability_mean", "class_separability_top10", "feat_outlier_frac", "eda_std_mean"]):
        ax.scatter(merged[col], merged["f1"])
        ax.set_xlabel(col)
        ax.set_ylabel("LOSO F1")
        c = merged[[col, "f1"]].corr().iloc[0, 1]
        ax.set_title(f"corr={c:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "f1_vs_stats.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
