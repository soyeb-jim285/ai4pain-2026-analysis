"""Tier-A4: Cluster subjects into "arm-vs-hand response types".

Hypothesis: pooled arm-vs-hand is at chance (0/646 features survive FDR,
classifiers ~0.54 macro-F1) because subjects respond to localisation
differently. If we can cluster subjects by baseline (NoPain) physiology and
find sub-groups within which ARM vs HAND IS distinguishable, then a
mixture-of-experts approach is worth building.

Pipeline
--------
1. Per-subject signature vectors  : mean feature vector across the 12
   NoPain segments per subject (train only). 41 subjects x 212 features.
   Standardise across subjects, PCA -> 10 components.
2. Cluster subjects with KMeans + Agglomerative for k = 2..5; silhouette
   on PCA space.
3. For each (method, k, cluster): paired Wilcoxon ARM vs HAND on
   subject-means, count nominal-significant + FDR-significant features,
   plus top Cliff's delta. Compare cluster-sum vs pooled.
4. Sub-group classifier (LOSO LogReg, subject-z features) within the most
   promising clustering, per cluster, vs pooled 0.54 baseline.
5. Characterise each 2-cluster solution: top-10 ANOVA-F features.
6. Plots + summary report.

Run
---
    uv run python scripts/16_tierA4_subject_clusters.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats as sstats  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from tqdm import tqdm  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ANALYSIS = Path(__file__).resolve().parents[1]
TAB_DIR = ANALYSIS / "results" / "tables"
REPORT_DIR = ANALYSIS / "results" / "reports"
PLOT_DIR = ANALYSIS / "plots" / "tierA4_clusters"
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
KS = (2, 3, 4, 5)
METHODS = ("kmeans", "agglo")
POOLED_LOSO_F1 = 0.54   # context: pooled arm-vs-hand val macro-F1


# ---------------------------------------------------------------------------
# Feature loading + cleanup
# ---------------------------------------------------------------------------
def load_clean_features() -> tuple[pd.DataFrame, list[str]]:
    """Read merged features, drop high-NaN/zero-var columns, median-impute."""
    df = pd.read_parquet(TAB_DIR / "all_features_merged.parquet")
    feat_cols = [c for c in df.columns if c not in META_COLS]
    nan_frac = df[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if nan_frac[c] <= 0.10]
    X = df[feat_cols].astype(np.float64)
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X), columns=feat_cols
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


# ---------------------------------------------------------------------------
# Subject signature + clustering
# ---------------------------------------------------------------------------
def per_subject_signature(
    df: pd.DataFrame, feat_cols: list[str]
) -> tuple[np.ndarray, list[int]]:
    """Mean feature vector per subject across NoPain (12) segments, train only."""
    base = df[(df["split"] == "train") & (df["class"] == "NoPain")]
    grp = base.groupby("subject", sort=True)[feat_cols].mean()
    return grp.values.astype(np.float64), list(grp.index)


def cluster_subjects(
    sig: np.ndarray, subjects: list[int], n_components: int = 10
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Standardise -> PCA -> KMeans/Agglomerative for k in KS.

    Returns:
        cluster_df  : one row per subject, columns subject + per-(method,k).
        sil_dict    : silhouette per (method, k).
        pca_coords  : (n_subj, n_components) PCA coordinates for plots/ANOVA.
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(sig)
    n_components = min(n_components, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    Z = pca.fit_transform(Xs)
    print(
        f"[pca] kept {n_components} components, "
        f"explained variance = {pca.explained_variance_ratio_.sum():.3f}"
    )

    rows = {"subject": subjects}
    sils: dict[tuple[str, int], float] = {}
    for k in KS:
        # KMeans
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(Z)
        rows[f"kmeans_k{k}"] = km.labels_
        try:
            sils[("kmeans", k)] = float(silhouette_score(Z, km.labels_))
        except Exception:
            sils[("kmeans", k)] = float("nan")
        # Agglomerative (ward)
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z)
        rows[f"agglo_k{k}"] = ag.labels_
        try:
            sils[("agglo", k)] = float(silhouette_score(Z, ag.labels_))
        except Exception:
            sils[("agglo", k)] = float("nan")

    return pd.DataFrame(rows), sils, Z


# ---------------------------------------------------------------------------
# Within-cluster arm-vs-hand stat tests
# ---------------------------------------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta between two 1-D arrays."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    # vectorised count of (a > b) and (a < b) pairs
    diffs = a[:, None] - b[None, :]
    gt = (diffs > 0).sum()
    lt = (diffs < 0).sum()
    return float((gt - lt) / (a.size * b.size))


def subject_class_means(
    df: pd.DataFrame, feat_cols: list[str], cls: str
) -> pd.DataFrame:
    """Mean feature vector per (subject, cls), train only."""
    sub = df[(df["split"] == "train") & (df["class"] == cls)]
    return sub.groupby("subject", sort=True)[feat_cols].mean()


def paired_wilcoxon_table(
    arm_means: pd.DataFrame, hand_means: pd.DataFrame, feat_cols: list[str],
    subjects: list[int],
) -> pd.DataFrame:
    """Paired Wilcoxon ARM vs HAND on subject-means for each feature."""
    arm = arm_means.reindex(subjects)
    hand = hand_means.reindex(subjects)
    rows = []
    for f in feat_cols:
        a = arm[f].to_numpy(dtype=float)
        h = hand[f].to_numpy(dtype=float)
        m = ~(np.isnan(a) | np.isnan(h))
        if m.sum() < 5:
            rows.append({"feature": f, "n_pairs": int(m.sum()),
                         "p_value": np.nan, "median_diff": np.nan,
                         "cliff_delta": np.nan})
            continue
        a, h = a[m], h[m]
        diff = a - h
        if np.allclose(diff, 0):
            p = 1.0
        else:
            try:
                _, p = sstats.wilcoxon(a, h, zero_method="wilcox",
                                       alternative="two-sided", mode="auto")
            except Exception:
                p = np.nan
        rows.append({
            "feature": f, "n_pairs": int(m.sum()),
            "p_value": float(p) if p is not None and np.isfinite(p) else np.nan,
            "median_diff": float(np.median(diff)),
            "cliff_delta": cliffs_delta(a, h),
        })
    out = pd.DataFrame(rows)
    valid = out["p_value"].notna()
    fdr = np.full(len(out), np.nan)
    if valid.any():
        try:
            _, q, _, _ = multipletests(out.loc[valid, "p_value"].values,
                                       alpha=0.05, method="fdr_bh")
            fdr[valid.values] = q
        except Exception:
            pass
    out["p_fdr"] = fdr
    return out


def summarise_test_table(
    test_df: pd.DataFrame, scheme: str, cluster_id: int, n_subjects: int
) -> dict:
    """Reduce a per-feature test table to a single row in the summary csv."""
    valid = test_df.dropna(subset=["p_value"])
    n_lt_01 = int((valid["p_value"] < 0.01).sum())
    n_fdr = int((valid["p_fdr"] < 0.05).sum()) if "p_fdr" in valid else 0
    if not valid.empty:
        # rank by |cliff delta|, ties broken by p
        ord_df = valid.assign(absd=valid["cliff_delta"].abs()).sort_values(
            ["absd", "p_value"], ascending=[False, True]
        )
        top = ord_df.iloc[0]
        top_feature = str(top["feature"])
        top_d = float(top["cliff_delta"])
        top_dir = "arm>hand" if top_d > 0 else ("hand>arm" if top_d < 0 else "n/a")
    else:
        top_feature, top_d, top_dir = "n/a", float("nan"), "n/a"
    return {
        "clustering_scheme": scheme,
        "cluster_id": int(cluster_id),
        "n_subjects": int(n_subjects),
        "n_p_lt_01": n_lt_01,
        "n_fdr_sig": n_fdr,
        "top_feature": top_feature,
        "top_cliff_delta": top_d,
        "top_direction": top_dir,
    }


# ---------------------------------------------------------------------------
# Within-cluster LOSO classifier
# ---------------------------------------------------------------------------
def apply_subject_z(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    means = out.groupby("subject")[feat_cols].transform("mean")
    stds = out.groupby("subject")[feat_cols].transform("std", ddof=0)
    stds = stds.where(stds > 0, 1.0)
    out[feat_cols] = (out[feat_cols] - means) / stds
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out


def cluster_loso(
    df_z: pd.DataFrame, feat_cols: list[str], cluster_subjects: list[int]
) -> dict:
    """LOSO LogReg ARM vs HAND restricted to a cluster's subjects."""
    if len(cluster_subjects) < 3:
        return {"n_subjects": len(cluster_subjects), "available": False,
                "macro_f1": float("nan"), "acc": float("nan"),
                "balanced_acc": float("nan"), "n_segments": 0}
    sub = df_z[
        (df_z["split"] == "train")
        & (df_z["subject"].isin(cluster_subjects))
        & (df_z["class"].isin(("PainArm", "PainHand")))
    ].reset_index(drop=True)
    if sub.empty:
        return {"n_subjects": len(cluster_subjects), "available": False,
                "macro_f1": float("nan"), "acc": float("nan"),
                "balanced_acc": float("nan"), "n_segments": 0}
    y = sub["class"].map({"PainArm": 0, "PainHand": 1}).to_numpy()
    X = sub[feat_cols].to_numpy()
    g = sub["subject"].to_numpy()
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups=g))
    f1s, accs, bals = [], [], []
    for tr, te in folds:
        sc = StandardScaler()
        try:
            X_tr = sc.fit_transform(X[tr])
            X_te = sc.transform(X[te])
            mdl = LogisticRegression(
                penalty="l2", C=1.0, class_weight="balanced",
                max_iter=3000, solver="lbfgs", random_state=SEED, n_jobs=1,
            )
            mdl.fit(X_tr, y[tr])
            yhat = mdl.predict(X_te)
        except Exception:
            continue
        f1s.append(f1_score(y[te], yhat, average="macro", zero_division=0))
        accs.append(accuracy_score(y[te], yhat))
        bals.append(balanced_accuracy_score(y[te], yhat))
    return {
        "n_subjects": len(cluster_subjects),
        "available": len(f1s) > 0,
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "macro_f1_std": float(np.std(f1s)) if f1s else float("nan"),
        "acc": float(np.mean(accs)) if accs else float("nan"),
        "balanced_acc": float(np.mean(bals)) if bals else float("nan"),
        "n_segments": int(len(sub)),
        "n_folds": len(f1s),
    }


# ---------------------------------------------------------------------------
# Cluster characterisation (which features distinguish the 2 clusters)
# ---------------------------------------------------------------------------
def characterise_clusters(
    sig: np.ndarray, subjects: list[int], feat_cols: list[str],
    labels: np.ndarray, scheme: str, top_n: int = 10,
) -> pd.DataFrame:
    """One-way ANOVA-F per feature across cluster labels (k=2 case)."""
    # standardise per feature for interpretable z-score deltas
    sc = StandardScaler()
    Z = sc.fit_transform(sig)
    rows = []
    uniq = np.unique(labels)
    for j, f in enumerate(feat_cols):
        groups = [Z[labels == c, j] for c in uniq]
        groups = [g for g in groups if g.size >= 2]
        if len(groups) < 2:
            continue
        try:
            F, p = sstats.f_oneway(*groups)
        except Exception:
            continue
        if not np.isfinite(F):
            continue
        means = {f"mean_z_cluster_{int(c)}": float(np.mean(Z[labels == c, j]))
                 for c in uniq}
        rows.append({
            "clustering_scheme": scheme,
            "feature": f,
            "F": float(F),
            "p_value": float(p),
            **means,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("F", ascending=False).head(top_n).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_pca_2d(
    Z: np.ndarray, subjects: list[int], labels: np.ndarray, scheme_name: str
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    palette = sns.color_palette("tab10", n_colors=int(labels.max()) + 1)
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=80, color=palette[int(c)],
                   edgecolor="black", linewidth=0.6,
                   label=f"cluster {int(c)} (n={int(m.sum())})")
    for i, sid in enumerate(subjects):
        ax.annotate(str(sid), (Z[i, 0], Z[i, 1]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points", alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"per-subject PCA — {scheme_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "subject_pca_2d_by_cluster.png", dpi=130)
    plt.close(fig)


def plot_signature_heatmap(
    sig: np.ndarray, subjects: list[int], feat_cols: list[str],
    labels: np.ndarray, top_features: list[str], scheme_name: str,
) -> None:
    sc = StandardScaler()
    Z = sc.fit_transform(sig)
    Zdf = pd.DataFrame(Z, index=subjects, columns=feat_cols)
    order = np.argsort(labels)
    Zord = Zdf.iloc[order]
    labels_ord = labels[order]
    top = [c for c in top_features if c in Zord.columns][:20]
    if not top:
        return
    mat = Zord[top]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.3 * len(mat))))
    sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-3, vmax=3,
                cbar_kws={"label": "z-score"}, ax=ax)
    # cluster boundaries
    boundaries = []
    cur = labels_ord[0]
    for i, l in enumerate(labels_ord):
        if l != cur:
            boundaries.append(i)
            cur = l
    for b in boundaries:
        ax.axhline(b, color="black", lw=2.0)
    ax.set_xlabel("top distinguishing features")
    ax.set_ylabel("subjects (sorted by cluster)")
    ax.set_title(f"signature heatmap — {scheme_name}")
    plt.xticks(rotation=70, fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cluster_signature_heatmap.png", dpi=130)
    plt.close(fig)


def plot_loso_bars(loso_df: pd.DataFrame) -> None:
    if loso_df.empty:
        return
    plot_df = loso_df[loso_df["available"]].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_df = plot_df.sort_values(
        ["clustering_scheme", "cluster_id"]
    ).reset_index(drop=True)
    xs = np.arange(len(plot_df))
    ax.bar(
        xs, plot_df["macro_f1"],
        color=["#6b8cae" if "kmeans" in s else "#cb997e"
               for s in plot_df["clustering_scheme"]],
        edgecolor="black", alpha=0.9,
    )
    ax.axhline(POOLED_LOSO_F1, color="#d62728", linestyle="--",
               label=f"pooled = {POOLED_LOSO_F1:.2f}")
    ax.axhline(0.5, color="grey", linestyle=":", label="chance = 0.50")
    ax.axhline(0.6, color="green", linestyle=":", label="target = 0.60")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r.clustering_scheme}\nc{r.cluster_id} (n={r.n_subjects})"
         for r in plot_df.itertuples()],
        fontsize=7, rotation=0,
    )
    ax.set_ylabel("LOSO macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("within-cluster LOSO macro-F1 vs pooled baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "within_cluster_macro_f1_bars.png", dpi=130)
    plt.close(fig)


def plot_nsig_bars(within_df: pd.DataFrame, pooled_n: int) -> None:
    if within_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    df = within_df.copy().reset_index(drop=True)
    xs = np.arange(len(df))
    ax.bar(
        xs, df["n_p_lt_01"],
        color=["#6b8cae" if "kmeans" in s else "#cb997e"
               for s in df["clustering_scheme"]],
        edgecolor="black", alpha=0.9,
    )
    ax.axhline(pooled_n, color="#d62728", linestyle="--",
               label=f"pooled = {pooled_n}")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r.clustering_scheme}\nc{r.cluster_id} (n={r.n_subjects})"
         for r in df.itertuples()],
        fontsize=6, rotation=70,
    )
    ax.set_ylabel("# features with paired p < 0.01")
    ax.set_title(
        "within-cluster nominal-significant arm-vs-hand features (uncorrected)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "within_cluster_nsig_bars.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("[load] all_features_merged.parquet")
    df, feat_cols = load_clean_features()
    print(f"[features] {len(feat_cols)} usable features")

    # 1. Per-subject signature
    sig, subjects = per_subject_signature(df, feat_cols)
    print(f"[signature] {sig.shape[0]} subjects x {sig.shape[1]} features")

    # 2. Cluster
    cluster_df, sils, Zpca = cluster_subjects(sig, subjects)
    cluster_df.to_csv(TAB_DIR / "tierA4_subject_clusters.csv", index=False)
    print("[save] tierA4_subject_clusters.csv")
    print("[silhouette]")
    for (m, k), s in sorted(sils.items()):
        print(f"  {m:7s} k={k}  silhouette={s:.3f}")

    # Pre-compute subject-class means once (train only)
    arm_means = subject_class_means(df, feat_cols, "PainArm")
    hand_means = subject_class_means(df, feat_cols, "PainHand")

    # Pooled baseline
    pooled_test = paired_wilcoxon_table(arm_means, hand_means, feat_cols, subjects)
    pooled_n_lt_01 = int((pooled_test["p_value"] < 0.01).sum())
    pooled_n_fdr = int((pooled_test["p_fdr"] < 0.05).sum())
    print(
        f"[pooled] paired Wilcoxon: n_p<0.01={pooled_n_lt_01}, "
        f"n_fdr<0.05={pooled_n_fdr} (across {len(subjects)} subjects)"
    )

    # 3. Within-cluster tests for every (method, k, cluster)
    within_rows: list[dict] = []
    cluster_sums: dict[tuple[str, int], int] = {}
    for method in METHODS:
        for k in KS:
            scheme = f"{method}_k{k}"
            labels = cluster_df[scheme].to_numpy()
            sum_n = 0
            for c in sorted(set(labels)):
                cl_subj = [s for s, l in zip(subjects, labels) if l == c]
                try:
                    test_df = paired_wilcoxon_table(
                        arm_means, hand_means, feat_cols, cl_subj
                    )
                    summary = summarise_test_table(test_df, scheme, c, len(cl_subj))
                except Exception as e:  # noqa: BLE001
                    summary = {
                        "clustering_scheme": scheme, "cluster_id": int(c),
                        "n_subjects": len(cl_subj),
                        "n_p_lt_01": 0, "n_fdr_sig": 0,
                        "top_feature": "n/a", "top_cliff_delta": float("nan"),
                        "top_direction": f"error:{e!s}",
                    }
                summary["silhouette"] = sils.get((method, k), float("nan"))
                summary["pooled_n_p_lt_01"] = pooled_n_lt_01
                summary["pooled_n_fdr_sig"] = pooled_n_fdr
                within_rows.append(summary)
                sum_n += summary["n_p_lt_01"]
            cluster_sums[(method, k)] = sum_n

    within_df = pd.DataFrame(within_rows)
    within_df.to_csv(TAB_DIR / "tierA4_within_cluster_tests.csv", index=False)
    print("[save] tierA4_within_cluster_tests.csv")

    # Pick the most promising (method, k) by max(cluster_sum - pooled).
    best_scheme = max(cluster_sums, key=lambda mk: cluster_sums[mk])
    best_method, best_k = best_scheme
    best_scheme_str = f"{best_method}_k{best_k}"
    print(
        f"[select] most promising scheme by cluster_sum n_p<0.01 = "
        f"{best_scheme_str} (sum={cluster_sums[best_scheme]}, "
        f"pooled={pooled_n_lt_01})"
    )

    # 4. Within-cluster LOSO classifier — best scheme only
    df_z = apply_subject_z(df, feat_cols)
    loso_rows: list[dict] = []
    labels_best = cluster_df[best_scheme_str].to_numpy()
    for c in sorted(set(labels_best)):
        cl_subj = [s for s, l in zip(subjects, labels_best) if l == c]
        try:
            res = cluster_loso(df_z, feat_cols, cl_subj)
        except Exception as e:  # noqa: BLE001
            res = {"available": False, "macro_f1": float("nan"),
                   "n_subjects": len(cl_subj), "n_segments": 0,
                   "error": str(e)}
        res.update({"clustering_scheme": best_scheme_str, "cluster_id": int(c)})
        loso_rows.append(res)

    # Also run for all (method, k) so the bar plot is meaningful.
    for method in METHODS:
        for k in KS:
            scheme = f"{method}_k{k}"
            if scheme == best_scheme_str:
                continue
            labels = cluster_df[scheme].to_numpy()
            for c in sorted(set(labels)):
                cl_subj = [s for s, l in zip(subjects, labels) if l == c]
                if len(cl_subj) < 5:
                    loso_rows.append({
                        "clustering_scheme": scheme, "cluster_id": int(c),
                        "n_subjects": len(cl_subj), "available": False,
                        "macro_f1": float("nan"), "n_segments": 0,
                    })
                    continue
                try:
                    res = cluster_loso(df_z, feat_cols, cl_subj)
                except Exception as e:  # noqa: BLE001
                    res = {"available": False, "macro_f1": float("nan"),
                           "n_subjects": len(cl_subj), "n_segments": 0,
                           "error": str(e)}
                res.update({"clustering_scheme": scheme, "cluster_id": int(c)})
                loso_rows.append(res)

    loso_df = pd.DataFrame(loso_rows)
    # ensure deterministic column order
    cols_order = [c for c in [
        "clustering_scheme", "cluster_id", "n_subjects", "n_segments",
        "available", "macro_f1", "macro_f1_std", "acc", "balanced_acc",
        "n_folds", "error",
    ] if c in loso_df.columns]
    loso_df = loso_df[cols_order]
    loso_df.to_csv(TAB_DIR / "tierA4_within_cluster_loso.csv", index=False)
    print("[save] tierA4_within_cluster_loso.csv")

    # 5. Characterise clusters: 2-cluster solution per method
    char_rows = []
    for method in METHODS:
        scheme = f"{method}_k2"
        try:
            char = characterise_clusters(
                sig, subjects, feat_cols,
                cluster_df[scheme].to_numpy(), scheme,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[characterise] {scheme} failed: {e}")
            continue
        char_rows.append(char)
    if char_rows:
        char_all = pd.concat(char_rows, ignore_index=True)
        char_all.to_csv(TAB_DIR / "tierA4_cluster_characteristics.csv", index=False)
        print("[save] tierA4_cluster_characteristics.csv")
    else:
        char_all = pd.DataFrame()

    # 6. Plots
    # PCA scatter — k=3 KMeans
    try:
        plot_pca_2d(Zpca, subjects, cluster_df["kmeans_k3"].to_numpy(),
                    "kmeans_k3")
    except Exception as e:  # noqa: BLE001
        print(f"[plot] pca scatter failed: {e}")

    # Signature heatmap — kmeans_k2 distinguishing features
    try:
        kmeans2_chars = (char_all[char_all["clustering_scheme"] == "kmeans_k2"]
                         if not char_all.empty else pd.DataFrame())
        top_feats = kmeans2_chars["feature"].tolist() if not kmeans2_chars.empty \
            else feat_cols[:20]
        plot_signature_heatmap(
            sig, subjects, feat_cols,
            cluster_df["kmeans_k2"].to_numpy(), top_feats, "kmeans_k2",
        )
    except Exception as e:  # noqa: BLE001
        print(f"[plot] signature heatmap failed: {e}")

    try:
        plot_loso_bars(loso_df)
    except Exception as e:  # noqa: BLE001
        print(f"[plot] loso bars failed: {e}")

    try:
        plot_nsig_bars(within_df, pooled_n_lt_01)
    except Exception as e:  # noqa: BLE001
        print(f"[plot] nsig bars failed: {e}")

    # 7. Report
    write_report(
        sils=sils, within_df=within_df, loso_df=loso_df,
        pooled_n_lt_01=pooled_n_lt_01, pooled_n_fdr=pooled_n_fdr,
        best_scheme=best_scheme_str, cluster_sums=cluster_sums,
        char_all=char_all, runtime=time.time() - t0,
        n_features=len(feat_cols), n_subjects=len(subjects),
    )
    print(f"[done] runtime {time.time() - t0:.1f}s")


def write_report(
    sils: dict, within_df: pd.DataFrame, loso_df: pd.DataFrame,
    pooled_n_lt_01: int, pooled_n_fdr: int, best_scheme: str,
    cluster_sums: dict, char_all: pd.DataFrame, runtime: float,
    n_features: int, n_subjects: int,
) -> None:
    lines = []
    lines.append("# Tier-A4: subject clustering for arm-vs-hand response types\n")
    lines.append(f"- subjects (train): **{n_subjects}**")
    lines.append(f"- usable features: **{n_features}**")
    lines.append(
        f"- pooled paired Wilcoxon ARM vs HAND: "
        f"**{pooled_n_lt_01}** features p<0.01, **{pooled_n_fdr}** FDR<0.05"
    )
    lines.append("")
    lines.append("## Silhouette per (method, k)")
    lines.append("")
    lines.append("| method | k | silhouette |")
    lines.append("|---|---|---|")
    for (m, k), s in sorted(sils.items()):
        lines.append(f"| {m} | {k} | {s:.3f} |")
    best_sil = max(sils, key=lambda mk: sils[mk])
    lines.append("")
    lines.append(
        f"Best silhouette: **{best_sil[0]} k={best_sil[1]} = "
        f"{sils[best_sil]:.3f}**"
    )

    lines.append("")
    lines.append("## Pooled vs cluster-sum n_p<0.01 (uncorrected)")
    lines.append("")
    lines.append("| scheme | cluster_sum n_p<0.01 | pooled | helps? |")
    lines.append("|---|---|---|---|")
    for (m, k), n in sorted(cluster_sums.items()):
        helps = "YES" if n > pooled_n_lt_01 else "no"
        lines.append(f"| {m}_k{k} | {n} | {pooled_n_lt_01} | {helps} |")
    lines.append("")
    lines.append(f"Most promising scheme by this metric: **{best_scheme}**")

    lines.append("")
    lines.append("## Within-cluster LOSO macro-F1 (best scheme)")
    lines.append("")
    lines.append(
        "Pooled arm-vs-hand baseline (context): "
        f"**{POOLED_LOSO_F1:.2f}** macro-F1"
    )
    lines.append("")
    lines.append(
        "| scheme | cluster | n_subj | n_seg | macro-F1 | acc | balanced acc |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    best_rows = loso_df[loso_df["clustering_scheme"] == best_scheme]
    for r in best_rows.itertuples():
        if not getattr(r, "available", False):
            lines.append(
                f"| {r.clustering_scheme} | {r.cluster_id} | {r.n_subjects} | "
                "n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            f"| {r.clustering_scheme} | {r.cluster_id} | {r.n_subjects} | "
            f"{getattr(r, 'n_segments', 0)} | {r.macro_f1:.3f} | "
            f"{getattr(r, 'acc', float('nan')):.3f} | "
            f"{getattr(r, 'balanced_acc', float('nan')):.3f} |"
        )

    # Win check across ALL schemes
    available = loso_df[loso_df["available"]] if "available" in loso_df else loso_df
    over_60 = (
        available[available["macro_f1"] > 0.60]
        if not available.empty else pd.DataFrame()
    )
    lines.append("")
    if over_60.empty:
        lines.append(
            "**No cluster reaches LOSO macro-F1 > 0.60** "
            "across any (method, k)."
        )
    else:
        lines.append("**Clusters reaching LOSO macro-F1 > 0.60:**")
        for r in over_60.itertuples():
            lines.append(
                f"- {r.clustering_scheme} cluster {r.cluster_id} "
                f"(n_subj={r.n_subjects}): macro-F1 = {r.macro_f1:.3f}"
            )

    # Top characterising features (k=2 schemes)
    lines.append("")
    lines.append("## Top features distinguishing the 2-cluster solutions")
    lines.append("")
    if not char_all.empty:
        for scheme in sorted(char_all["clustering_scheme"].unique()):
            lines.append(f"**{scheme}**:")
            sub = char_all[char_all["clustering_scheme"] == scheme]
            for r in sub.itertuples():
                lines.append(f"- {r.feature}  (F={r.F:.2f}, p={r.p_value:.2g})")
            lines.append("")

    # Verdict
    lines.append("## Verdict")
    win_loso = (not over_60.empty)
    win_nsig = any(n > pooled_n_lt_01 for n in cluster_sums.values())
    if win_loso and win_nsig:
        verdict = (
            "**Mixture-of-experts is worth building.** Within-cluster "
            "macro-F1 exceeds 0.60 in at least one cluster AND clustering "
            "concentrates more nominal-significant features than the pooled "
            "test does."
        )
    elif win_nsig and not win_loso:
        verdict = (
            "**Mixed signal.** Clustering increases the count of nominal-"
            "significant arm-vs-hand features above the pooled baseline, "
            "suggesting heterogeneity exists, but no single sub-group "
            "classifier crosses macro-F1 = 0.60. MoE is worth a small-"
            "scale prototype but not a major investment."
        )
    elif win_loso and not win_nsig:
        verdict = (
            "**Possible win on classifier alone.** A cluster reaches "
            "macro-F1 > 0.60 even though the per-feature significance does "
            "not concentrate. Inspect that cluster carefully — could be a "
            "lucky small sample. MoE prototype is justified only if the "
            "winning cluster is replicated."
        )
    else:
        verdict = (
            "**MoE is NOT worth it on this signature.** Neither the per-"
            "feature significance count nor any within-cluster classifier "
            "beats the pooled baseline meaningfully. The localisation "
            "signal probably isn't separable along the baseline-physiology "
            "axis we clustered on."
        )
    lines.append(verdict)

    lines.append("")
    lines.append(f"_runtime: {runtime:.1f}s_")

    fp = REPORT_DIR / "16_tierA4_subject_clusters_summary.md"
    fp.write_text("\n".join(lines))
    print(f"[save] {fp}")


if __name__ == "__main__":
    main()
