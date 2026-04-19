"""Tier A #5 — Channel-restricted ARM-vs-HAND analysis.

Isolate which single physiological channel (BVP / EDA / RESP / SpO2)
carries the arm-vs-hand signal by:
  1. tagging each feature with its source channel,
  2. running paired Wilcoxon (subject-mean) restricted to each channel,
  3. running LOSO classifiers restricted to each channel,
  4. building a feature-count vs macro-F1 curve per channel.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sstats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
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
PLOT_DIR = ANALYSIS / "plots" / "tierA5_channel"
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
ARM_HAND = ("PainArm", "PainHand")
LABEL_MAP = {"PainArm": 0, "PainHand": 1}
SIGNAL_TAGS = {"bvp": ["bvp"], "eda": ["eda"], "resp": ["resp"], "spo2": ["spo2", "spo_2"]}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diffs = a[:, None] - b[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / (a.size * b.size))


def classify_feature(name: str) -> str:
    n = name.lower()
    hits = []
    for tag, kws in SIGNAL_TAGS.items():
        if any(k in n for k in kws):
            hits.append(tag)
    # de-dupe (resp can match in 'resp' too)
    hits = sorted(set(hits))
    if not hits:
        return "other"
    if len(hits) == 1:
        return hits[0]
    return "cross"


# ---------------------------------------------------------------------------
def main() -> None:
    fp = TAB_DIR / "all_features_merged.parquet"
    if not fp.exists():
        raise SystemExit(f"missing {fp}")
    df = pd.read_parquet(fp)

    # Filter to PainArm + PainHand only
    df = df[df["class"].isin(ARM_HAND)].copy().reset_index(drop=True)
    feat_cols_all = [c for c in df.columns if c not in META_COLS]
    nan_frac = df[feat_cols_all].isna().mean()
    feat_cols_all = [c for c in feat_cols_all if nan_frac[c] <= 0.10]
    X = df[feat_cols_all].astype(np.float64)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                     columns=feat_cols_all)
    var = X.var(axis=0)
    feat_cols_all = var.index[var > 1e-12].tolist()
    X = X[feat_cols_all].astype(np.float32)
    df = pd.concat([df[META_COLS].reset_index(drop=True),
                    X.reset_index(drop=True)], axis=1)
    print(f"[load] {len(feat_cols_all)} features × {len(df)} arm+hand segments")

    # 1. Channel assignment
    chan_map = pd.DataFrame(
        [{"feature": c, "channel": classify_feature(c)} for c in feat_cols_all]
    )
    chan_map.to_csv(TAB_DIR / "tierA5_feature_channel_map.csv", index=False)
    print("[map] channel counts:")
    print(chan_map["channel"].value_counts())

    # Subject-z normalisation across all 36 segments per subject
    # (we only kept ARM+HAND but groupby subject still works)
    means = df.groupby("subject")[feat_cols_all].transform("mean")
    stds = df.groupby("subject")[feat_cols_all].transform("std", ddof=0)
    stds = stds.where(stds > 0, 1.0)
    df[feat_cols_all] = ((df[feat_cols_all] - means) / stds).fillna(0.0).astype(np.float32)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)

    # 2. Per-channel paired Wilcoxon (subject-means)
    print("\n[2] per-channel paired Wilcoxon ...")
    test_rows = []
    subj_means_arm = train[train["class"] == "PainArm"].groupby("subject")[feat_cols_all].mean()
    subj_means_hand = train[train["class"] == "PainHand"].groupby("subject")[feat_cols_all].mean()
    common_subjects = subj_means_arm.index.intersection(subj_means_hand.index)
    subj_means_arm = subj_means_arm.loc[common_subjects]
    subj_means_hand = subj_means_hand.loc[common_subjects]
    n_subj = len(common_subjects)
    for feat in tqdm(feat_cols_all, desc="wilcoxon"):
        a = subj_means_arm[feat].to_numpy()
        h = subj_means_hand[feat].to_numpy()
        valid = ~(np.isnan(a) | np.isnan(h))
        if valid.sum() < 5:
            test_rows.append({"feature": feat, "channel": classify_feature(feat),
                              "n": int(valid.sum()), "W": np.nan, "p": np.nan,
                              "cliff": np.nan, "sign_arm_gt_hand": np.nan})
            continue
        a, h = a[valid], h[valid]
        try:
            res = sstats.wilcoxon(a, h, zero_method="wilcox",
                                  alternative="two-sided", mode="auto")
            W, p = float(res.statistic), float(res.pvalue)
        except Exception:
            W, p = np.nan, np.nan
        d = cliffs_delta(a, h)
        sgn = float(np.mean(a > h))
        test_rows.append({"feature": feat, "channel": classify_feature(feat),
                          "n": int(a.size), "W": W, "p": p, "cliff": d,
                          "sign_arm_gt_hand": sgn})
    tests = pd.DataFrame(test_rows)
    # FDR per channel family
    tests["p_fdr"] = np.nan
    for ch in tests["channel"].unique():
        idx = tests.index[(tests["channel"] == ch) & tests["p"].notna()]
        if len(idx) == 0:
            continue
        _, p_corr, _, _ = multipletests(tests.loc[idx, "p"].values,
                                        method="fdr_bh")
        tests.loc[idx, "p_fdr"] = p_corr
    tests.to_csv(TAB_DIR / "tierA5_per_channel_tests.csv", index=False)

    summary_per_chan = (
        tests.assign(sig05=(tests["p_fdr"] < 0.05).astype(int),
                     sig10=(tests["p_fdr"] < 0.10).astype(int),
                     nominal_sig=(tests["p"] < 0.05).astype(int))
        .groupby("channel")[["sig05", "sig10", "nominal_sig", "feature"]]
        .agg({"sig05": "sum", "sig10": "sum", "nominal_sig": "sum",
              "feature": "count"})
        .rename(columns={"feature": "n_features"})
        .reset_index()
        .sort_values("sig05", ascending=False)
    )
    summary_per_chan.to_csv(TAB_DIR / "tierA5_per_channel_summary.csv", index=False)
    print("\n[2] per-channel significance summary:")
    print(summary_per_chan)

    # 3. Per-channel LOSO classifier
    print("\n[3] per-channel LOSO classifier ...")
    channels = ["bvp", "eda", "resp", "spo2", "cross", "all"]
    models = {"logreg": lambda: LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced", max_iter=3000,
        solver="lbfgs", n_jobs=1, random_state=SEED,
    ), "rf": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=SEED,
    )}
    if _HAS_XGB:
        models["xgb"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            max_bin=128, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=4, verbosity=0,
        )

    loso_rows = []
    subjects_train = train["subject"].to_numpy()
    y_train = train["class"].map(LABEL_MAP).to_numpy()
    y_val = val["class"].map(LABEL_MAP).to_numpy() if not val.empty else None

    for ch in channels:
        if ch == "all":
            cols = feat_cols_all
        else:
            cols = chan_map[chan_map["channel"] == ch]["feature"].tolist()
        if len(cols) < 3:
            print(f"  skip {ch}: only {len(cols)} features")
            continue
        Xt = train[cols].to_numpy()
        Xv = val[cols].to_numpy() if not val.empty else None
        for mname, mfac in models.items():
            print(f"  channel={ch:<6} model={mname:<7} n_feat={len(cols)}")
            logo = LeaveOneGroupOut()
            f1s = []
            accs = []
            baccs = []
            for tr, te in logo.split(Xt, y_train, groups=subjects_train):
                sc = StandardScaler().fit(Xt[tr])
                mdl = mfac()
                mdl.fit(sc.transform(Xt[tr]), y_train[tr])
                yhat = mdl.predict(sc.transform(Xt[te]))
                f1s.append(f1_score(y_train[te], yhat, average="macro",
                                    zero_division=0))
                accs.append(accuracy_score(y_train[te], yhat))
                baccs.append(balanced_accuracy_score(y_train[te], yhat))
            row = {
                "channel": ch, "model": mname, "n_features": len(cols),
                "loso_macro_f1_mean": float(np.mean(f1s)),
                "loso_macro_f1_std": float(np.std(f1s)),
                "loso_acc_mean": float(np.mean(accs)),
                "loso_balanced_acc_mean": float(np.mean(baccs)),
                "n_folds": len(f1s),
            }
            if Xv is not None:
                sc = StandardScaler().fit(Xt)
                mdl = mfac()
                mdl.fit(sc.transform(Xt), y_train)
                yhat_v = mdl.predict(sc.transform(Xv))
                row["val_macro_f1"] = float(f1_score(y_val, yhat_v,
                                                    average="macro",
                                                    zero_division=0))
                row["val_acc"] = float(accuracy_score(y_val, yhat_v))
                row["val_balanced_acc"] = float(balanced_accuracy_score(y_val, yhat_v))
            loso_rows.append(row)
    loso_df = pd.DataFrame(loso_rows)
    loso_df.to_csv(TAB_DIR / "tierA5_per_channel_loso.csv", index=False)

    # 4. Feature-count curve per channel (LR only, for speed)
    print("\n[4] feature-count curve ...")
    curve_rows = []
    Ks = [5, 10, 20, 40, 80, 160]
    for ch in channels:
        if ch == "all":
            cols_all = feat_cols_all
        else:
            cols_all = chan_map[chan_map["channel"] == ch]["feature"].tolist()
        if len(cols_all) < 5:
            continue
        # rank features by |Cliff's δ| within this channel
        sub = tests[tests["feature"].isin(cols_all)].copy()
        sub["abs_cliff"] = sub["cliff"].abs()
        sub = sub.sort_values("abs_cliff", ascending=False)
        ranked = sub["feature"].tolist()
        for K in Ks + ["all"]:
            if K == "all":
                cols = ranked
            else:
                if K > len(ranked):
                    continue
                cols = ranked[:K]
            Xt = train[cols].to_numpy()
            logo = LeaveOneGroupOut()
            f1s = []
            for tr, te in logo.split(Xt, y_train, groups=subjects_train):
                sc = StandardScaler().fit(Xt[tr])
                mdl = LogisticRegression(
                    penalty="l2", C=1.0, class_weight="balanced",
                    max_iter=3000, solver="lbfgs", random_state=SEED, n_jobs=1,
                )
                mdl.fit(sc.transform(Xt[tr]), y_train[tr])
                yhat = mdl.predict(sc.transform(Xt[te]))
                f1s.append(f1_score(y_train[te], yhat, average="macro",
                                    zero_division=0))
            curve_rows.append({
                "channel": ch, "K": K if K != "all" else len(cols),
                "K_label": str(K),
                "loso_macro_f1_mean": float(np.mean(f1s)),
                "loso_macro_f1_std": float(np.std(f1s)),
                "n_features_used": len(cols),
            })
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(TAB_DIR / "tierA5_feature_count_curve.csv", index=False)

    # 5. Plots
    import matplotlib.pyplot as plt
    import seaborn as sns

    # n_significant per channel
    fig, ax = plt.subplots(figsize=(8, 4))
    s = summary_per_chan.set_index("channel")[["sig05", "sig10", "nominal_sig"]]
    s.plot(kind="bar", ax=ax)
    ax.set_ylabel("n features")
    ax.set_title("Per-channel ARM vs HAND significance counts")
    ax.legend(["FDR<0.05", "FDR<0.10", "raw p<0.05"])
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "per_channel_nsig_bar.png", dpi=130)
    plt.close(fig)

    # top Cliff's delta per channel
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_rows = []
    for ch in tests["channel"].unique():
        top = (tests[tests["channel"] == ch]
               .assign(abs_cliff=tests["cliff"].abs())
               .sort_values("abs_cliff", ascending=False)
               .head(5))
        for _, r in top.iterrows():
            plot_rows.append({"channel": ch, "feature": r["feature"][-30:],
                              "cliff": r["cliff"]})
    plot_df = pd.DataFrame(plot_rows)
    if not plot_df.empty:
        sns.barplot(data=plot_df, x="cliff", y="feature", hue="channel", ax=ax)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title("Top 5 features per channel by |Cliff's δ|")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "per_channel_top_cliff_delta.png", dpi=130)
    plt.close(fig)

    # macro-F1 grouped bar
    if not loso_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        pivot = loso_df.pivot(index="channel", columns="model",
                              values="loso_macro_f1_mean")
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0.5, color="r", linestyle="--", label="chance")
        ax.set_ylabel("LOSO macro-F1")
        ax.set_ylim(0.3, 0.7)
        ax.set_title("Per-channel LOSO macro-F1")
        ax.legend()
        plt.xticks(rotation=20)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "per_channel_macro_f1_grouped_bar.png", dpi=130)
        plt.close(fig)

    # feature-count curve
    if not curve_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ch in curve_df["channel"].unique():
            sub = curve_df[curve_df["channel"] == ch].sort_values("K")
            ax.plot(sub["K"], sub["loso_macro_f1_mean"], marker="o", label=ch)
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("top-K features")
        ax.set_ylabel("LOSO macro-F1")
        ax.set_xscale("log")
        ax.set_title("Feature-count vs LOSO macro-F1, per channel")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "feature_count_curve_per_channel.png", dpi=130)
        plt.close(fig)

    # Report
    lines = ["# Tier A #5 — Channel-restricted ARM vs HAND\n"]
    lines.append(f"- {len(feat_cols_all)} features after NaN/variance filter")
    lines.append(f"- {n_subj} train subjects with both ARM and HAND segments\n")
    lines.append("## Per-channel significance counts (paired Wilcoxon, BH-FDR)\n")
    lines.append(summary_per_chan.to_markdown(index=False))
    lines.append("")
    lines.append("## Per-channel LOSO macro-F1\n")
    if not loso_df.empty:
        lines.append(loso_df.sort_values("loso_macro_f1_mean", ascending=False)
                     .to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Feature-count vs macro-F1 curve\n")
    if not curve_df.empty:
        lines.append(curve_df.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Top 10 features overall by smallest p\n")
    top10 = tests.sort_values("p", na_position="last").head(10)
    lines.append(top10[["channel", "feature", "p", "p_fdr",
                        "cliff", "sign_arm_gt_hand"]]
                 .to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    best_chan = (loso_df.sort_values("loso_macro_f1_mean", ascending=False)
                 .iloc[0] if not loso_df.empty else None)
    lines.append("## Verdict\n")
    if best_chan is not None:
        lines.append(
            f"- Best single channel: **{best_chan['channel']}** "
            f"(model={best_chan['model']}, "
            f"LOSO macro-F1={best_chan['loso_macro_f1_mean']:.3f}, "
            f"val={best_chan.get('val_macro_f1', float('nan')):.3f})."
        )
    pooled = (loso_df[loso_df["channel"] == "all"]["loso_macro_f1_mean"].max()
              if not loso_df.empty else float("nan"))
    lines.append(f"- Pooled (all-channels) best LOSO macro-F1: {pooled:.3f}")
    lines.append("")
    rep_fp = REPORT_DIR / "17_tierA5_channel_restricted_summary.md"
    rep_fp.write_text("\n".join(lines))
    print(f"\n[save] {rep_fp}")
    print("Done.")


if __name__ == "__main__":
    main()
