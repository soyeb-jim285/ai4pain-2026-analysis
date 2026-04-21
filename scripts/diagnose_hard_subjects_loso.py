"""
Diagnose why bottom-10 LOSO subjects perform poorly.

Compares bottom-10 vs top-10 LOSO subjects (from combined_best per_subject.csv)
on four hypotheses:
  H1. Weak physio response  : NoPain->Pain amplitude delta small
  H2. Weak arm-vs-hand signal: Cliff's delta(Arm vs Hand) on resp_all small
  H3. Low signal quality     : per-subject NaN frac / feature std
  H4. Feature outlier        : per-subject feature z-scores vs cohort mean

Outputs:
  results/final/stage2_hard_subjects/hard_vs_easy_report.md
  results/final/stage2_hard_subjects/per_subject_stats.csv
  plots/stage2_hard_subjects/*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.final_pipeline import (
    ARM_HAND,
    channel_of,
    cliffs_delta,
    load_clean_features,
    stage2_feature_sets,
    unique_by_canonical,
)


def per_subject_stats(df_all: pd.DataFrame, feat_cols: list[str], resp_feats: list[str]) -> pd.DataFrame:
    rows = []
    for subj, sub in df_all.groupby("subject"):
        nopain = sub[sub["class"] == "NoPain"][feat_cols]
        pain = sub[sub["class"].isin(ARM_HAND)][feat_cols]
        arm = sub[sub["class"] == "PainArm"][resp_feats]
        hand = sub[sub["class"] == "PainHand"][resp_feats]

        # H1: NoPain -> Pain amplitude delta (reactivity)
        if not nopain.empty and not pain.empty:
            diff = (pain.mean(axis=0) - nopain.mean(axis=0)).abs()
            reactivity_l2 = float(np.linalg.norm(diff.fillna(0.0).to_numpy()))
            reactivity_mean = float(diff.mean())
        else:
            reactivity_l2 = reactivity_mean = np.nan

        # H2: Arm vs Hand discriminability via Cliff's delta on resp features
        cliffs = []
        for feat in resp_feats:
            a = arm[feat].dropna().to_numpy()
            b = hand[feat].dropna().to_numpy()
            if a.size >= 3 and b.size >= 3:
                cliffs.append(abs(cliffs_delta(a, b)))
        arm_hand_sep = float(np.mean(cliffs)) if cliffs else np.nan
        arm_hand_sep_max = float(np.max(cliffs)) if cliffs else np.nan

        # H3: signal quality
        nan_frac = float(sub[feat_cols].isna().mean().mean())
        within_std = float(sub[feat_cols].std(axis=0, ddof=0).mean())
        within_iqr = float((sub[feat_cols].quantile(0.75) - sub[feat_cols].quantile(0.25)).mean())

        rows.append({
            "subject": int(subj),
            "split": str(sub["split"].iloc[0]),
            "reactivity_l2": reactivity_l2,
            "reactivity_mean": reactivity_mean,
            "arm_hand_sep_mean": arm_hand_sep,
            "arm_hand_sep_max": arm_hand_sep_max,
            "nan_frac": nan_frac,
            "within_std_mean": within_std,
            "within_iqr_mean": within_iqr,
        })
    return pd.DataFrame(rows)


def compare_groups(stats: pd.DataFrame, easy: list[int], hard: list[int], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        h = stats[stats["subject"].isin(hard)][m].to_numpy()
        e = stats[stats["subject"].isin(easy)][m].to_numpy()
        h = h[~np.isnan(h)]
        e = e[~np.isnan(e)]
        if h.size == 0 or e.size == 0:
            continue
        # Cliff's delta easy vs hard (positive = easy > hard)
        diffs = e[:, None] - h[None, :]
        cd = float((np.sum(diffs > 0) - np.sum(diffs < 0)) / (e.size * h.size))
        rows.append({
            "metric": m,
            "hard_mean": float(h.mean()),
            "hard_median": float(np.median(h)),
            "easy_mean": float(e.mean()),
            "easy_median": float(np.median(e)),
            "delta_mean": float(e.mean() - h.mean()),
            "cliffs_delta_easy_vs_hard": cd,
        })
    return pd.DataFrame(rows)


def plot_distributions(stats: pd.DataFrame, easy: list[int], hard: list[int],
                        metrics: list[str], out_path: Path) -> None:
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        h = stats[stats["subject"].isin(hard)][m].dropna().to_numpy()
        e = stats[stats["subject"].isin(easy)][m].dropna().to_numpy()
        other = stats[~stats["subject"].isin(hard + easy)][m].dropna().to_numpy()
        ax.boxplot([h, other, e], labels=["hard-10", "mid", "easy-10"], widths=0.6)
        ax.scatter(np.ones(len(h)) * 1, h, alpha=0.6, color="tab:red", zorder=3)
        ax.scatter(np.ones(len(other)) * 2, other, alpha=0.3, color="gray", zorder=3)
        ax.scatter(np.ones(len(e)) * 3, e, alpha=0.6, color="tab:green", zorder=3)
        ax.set_title(m, fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    per_subj_path = Path(args.per_subject_csv)
    if not per_subj_path.exists():
        raise SystemExit(f"missing {per_subj_path} -- run scripts/run_combined.py first")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    per_subj = pd.read_csv(per_subj_path)
    loso = per_subj[per_subj["split"] == "train_loso"].sort_values("macro_f1").reset_index(drop=True)
    hard = loso.head(args.k)["subject"].astype(int).tolist()
    easy = loso.tail(args.k)["subject"].astype(int).tolist()
    print(f"[loso] n={len(loso)} hard-{args.k}={hard} easy-{args.k}={easy}")
    print(f"[loso] hard mean_f1={loso.head(args.k).macro_f1.mean():.4f} "
          f"easy mean_f1={loso.tail(args.k).macro_f1.mean():.4f}")

    df_all, feat_cols = load_clean_features(args.feature_parquet)
    train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
    feature_sets = stage2_feature_sets(train_df, feat_cols)
    resp_feats = feature_sets[args.feature_set]

    print(f"[diag] computing stats for {df_all['subject'].nunique()} subjects, "
          f"{len(feat_cols)} feats, {len(resp_feats)} resp feats")
    stats = per_subject_stats(df_all, feat_cols, resp_feats)
    stats["loso_macro_f1"] = stats["subject"].map(
        loso.set_index("subject")["macro_f1"].astype(float).to_dict()
    )
    stats["group"] = np.where(stats["subject"].isin(hard), "hard",
                              np.where(stats["subject"].isin(easy), "easy", "mid"))
    stats.to_csv(out_dir / "per_subject_stats.csv", index=False)

    metrics = ["reactivity_l2", "reactivity_mean", "arm_hand_sep_mean",
                "arm_hand_sep_max", "nan_frac", "within_std_mean"]
    comp = compare_groups(stats, easy, hard, metrics)
    print("\n=== Hard-10 vs Easy-10 ===")
    print(comp.to_string(index=False))
    comp.to_csv(out_dir / "easy_vs_hard_comparison.csv", index=False)

    plot_distributions(stats, easy, hard, metrics,
                        plot_dir / "hard_vs_easy_boxplots.png")

    # Correlation between each metric and per-subject LOSO F1 (train only)
    corr_rows = []
    train_stats = stats[stats["split"] == "train"].copy()
    for m in metrics:
        mask = train_stats[m].notna() & train_stats["loso_macro_f1"].notna()
        if mask.sum() < 5:
            continue
        r = float(np.corrcoef(train_stats.loc[mask, m], train_stats.loc[mask, "loso_macro_f1"])[0, 1])
        corr_rows.append({"metric": m, "pearson_r_vs_loso_f1": r, "n": int(mask.sum())})
    corr = pd.DataFrame(corr_rows).sort_values("pearson_r_vs_loso_f1", ascending=False)
    print("\n=== Correlation with LOSO macro-F1 (n={}) ===".format(len(train_stats)))
    print(corr.to_string(index=False))
    corr.to_csv(out_dir / "metric_vs_loso_correlation.csv", index=False)

    # Scatter: best-correlating metric vs LOSO F1
    if not corr.empty:
        top_metric = corr.iloc[0]["metric"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for grp, color in (("hard", "tab:red"), ("mid", "gray"), ("easy", "tab:green")):
            sub = train_stats[train_stats["group"] == grp]
            ax.scatter(sub[top_metric], sub["loso_macro_f1"], alpha=0.7, color=color, label=f"{grp} (n={len(sub)})")
        ax.set_xlabel(top_metric)
        ax.set_ylabel("LOSO macro-F1")
        ax.set_title(f"LOSO F1 vs {top_metric} (Pearson r={corr.iloc[0]['pearson_r_vs_loso_f1']:.3f})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"scatter_{top_metric}_vs_loso_f1.png", dpi=110)
        plt.close(fig)

    # Modality breakdown: arm-vs-hand separability per modality for hard/easy
    mod_rows = []
    for mod in ("bvp", "eda", "resp"):
        mod_feats = unique_by_canonical([c for c in feat_cols if channel_of(c) == mod])
        if not mod_feats:
            continue
        for subj in stats["subject"]:
            sub = df_all[df_all["subject"] == subj]
            arm = sub[sub["class"] == "PainArm"][mod_feats]
            hand = sub[sub["class"] == "PainHand"][mod_feats]
            cds = []
            for feat in mod_feats:
                a = arm[feat].dropna().to_numpy()
                b = hand[feat].dropna().to_numpy()
                if a.size >= 3 and b.size >= 3:
                    cds.append(abs(cliffs_delta(a, b)))
            mean_cd = float(np.mean(cds)) if cds else np.nan
            grp = stats.set_index("subject").loc[subj, "group"]
            mod_rows.append({"subject": int(subj), "modality": mod,
                              "mean_arm_hand_cliffs": mean_cd, "group": grp})
    mod_df = pd.DataFrame(mod_rows)
    mod_summary = mod_df.groupby(["group", "modality"])["mean_arm_hand_cliffs"].agg(["mean", "median", "count"]).reset_index()
    print("\n=== Per-modality Arm-vs-Hand separability by group ===")
    print(mod_summary.to_string(index=False))
    mod_summary.to_csv(out_dir / "per_modality_separability.csv", index=False)

    # Report
    report = ["# Hard-Subject Diagnosis", "",
              f"- k = {args.k} (hard = bottom-k by LOSO macro-F1, easy = top-k)",
              f"- hard subjects: {hard}",
              f"- easy subjects: {easy}",
              f"- hard mean F1: {loso.head(args.k).macro_f1.mean():.4f}",
              f"- easy mean F1: {loso.tail(args.k).macro_f1.mean():.4f}",
              "", "## Group comparison", "",
              comp.to_markdown(index=False, floatfmt=".4f"),
              "", "## Metric ↔ LOSO F1 correlation", "",
              corr.to_markdown(index=False, floatfmt=".4f"),
              "", "## Per-modality arm-vs-hand separability", "",
              mod_summary.to_markdown(index=False, floatfmt=".4f"),
              "", "## Interpretation guide", "",
              "- High `|r|` with LOSO F1 + clear hard/easy gap = that metric drives poor performance.",
              "- If `arm_hand_sep_mean` is the top driver: hard subjects genuinely lack Arm-vs-Hand signal.",
              "- If `reactivity_l2` dominates: hard subjects are weak responders (low NoPain->Pain delta).",
              "- If `nan_frac` or `within_std_mean` dominate: signal-quality / sensor issue.",
              "- If no single metric explains it: mixed causes, may need per-subject modelling.",
              ]
    (out_dir / "hard_vs_easy_report.md").write_text("\n".join(report))
    print(f"\n[done] wrote {out_dir}/ and {plot_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-subject-csv", default="results/final/combined_best/per_subject.csv")
    p.add_argument("--feature-parquet", default="results/tables/all_features_merged_1022.parquet")
    p.add_argument("--feature-set", default="resp_all",
                    choices=["resp_all", "resp_top20", "resp_bvp5",
                             "eda_resp_top30", "bvp_resp_top30", "all_top40"])
    p.add_argument("--k", type=int, default=10, help="hard/easy group size")
    p.add_argument("--output-dir", default="results/final/stage2_hard_subjects")
    p.add_argument("--plot-dir", default="plots/stage2_hard_subjects")
    main(p.parse_args())
