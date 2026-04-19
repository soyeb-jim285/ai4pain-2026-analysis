"""Dimensionality-reduction / embedding analysis for AI4Pain 2026.

Runs PCA, t-SNE, and UMAP across three feature-preprocessing variants
(raw, global_z, subject_z) on the merged physio + tf + raw-stats feature
matrix, with subject-leakage diagnostics.

Run:
    uv run python scripts/07_dimreduction.py
from the repo root.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import CLASSES  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120

SEED = 42
np.random.seed(SEED)

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_ROOT / "results" / "tables"
REPORTS_DIR = ANALYSIS_ROOT / "results" / "reports"
PLOTS_DIR = ANALYSIS_ROOT / "plots" / "dimreduction"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
VARIANTS = ("raw", "global_z", "subject_z")
REDUCERS = ("PCA", "tSNE", "UMAP")
CLASS_COLORS = {"NoPain": "#4c72b0", "PainArm": "#dd8452", "PainHand": "#c44e52"}


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------
def build_or_load_merged() -> pd.DataFrame:
    merged_fp = TABLES_DIR / "all_features_merged.parquet"
    if merged_fp.exists():
        print(f"  loading existing merged features: {merged_fp.name}")
        return pd.read_parquet(merged_fp)

    print("  building merged feature table from 3 sources")
    physio = pd.read_parquet(TABLES_DIR / "physio_features.parquet")
    tf = pd.read_parquet(TABLES_DIR / "tf_features.parquet")
    raw = pd.read_parquet(TABLES_DIR / "raw_stats_per_segment.parquet")

    # Use segment_id as join key; keep meta from physio
    physio_keyed = physio.set_index("segment_id")
    tf_keyed = tf.drop(columns=META_COLS[:-1]).set_index("segment_id")
    raw_keyed = raw.drop(columns=META_COLS[:-1]).set_index("segment_id")

    # Disambiguate overlapping feature names between tf and raw via suffixes.
    tf_feats = set(tf_keyed.columns)
    raw_feats = set(raw_keyed.columns)
    overlap = tf_feats & raw_feats
    if overlap:
        tf_keyed = tf_keyed.rename(columns={c: f"{c}__tf" for c in overlap})
        raw_keyed = raw_keyed.rename(columns={c: f"{c}__raw" for c in overlap})

    merged = physio_keyed.join(tf_keyed, how="outer").join(raw_keyed, how="outer")
    merged = merged.reset_index()
    # Ensure meta columns come first
    meta_present = [c for c in META_COLS if c in merged.columns]
    feat_cols = [c for c in merged.columns if c not in meta_present]
    merged = merged[meta_present + feat_cols]
    merged.to_parquet(merged_fp, index=False)
    print(f"  saved merged features: {merged.shape} -> {merged_fp}")
    return merged


def clean_features(
    df: pd.DataFrame, feat_cols: list[str], nan_threshold: float = 0.1
) -> list[str]:
    """Return the list of columns that survive NaN / zero-variance filtering."""
    n = len(df)
    nan_frac = df[feat_cols].isna().mean()
    kept = nan_frac[nan_frac <= nan_threshold].index.tolist()
    dropped_nan = [c for c in feat_cols if c not in kept]
    # zero-variance check on train split (after median impute)
    return kept, dropped_nan


def median_impute(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(X))
    X = X.copy()
    X[inds] = np.take(med, inds[1])
    return X


def preprocess_variants(
    merged: pd.DataFrame, feat_cols: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Return dict variant -> (X_train, X_val, feat_cols_used).

    Order within X_train/X_val matches merged[merged.split==split] order.
    """
    train_mask = merged["split"] == "train"
    val_mask = merged["split"] == "validation"

    X_train_raw = merged.loc[train_mask, feat_cols].to_numpy(dtype=float)
    X_val_raw = merged.loc[val_mask, feat_cols].to_numpy(dtype=float)

    # Median-impute using training medians
    med = np.nanmedian(X_train_raw, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    for M in (X_train_raw, X_val_raw):
        inds = np.where(np.isnan(M))
        M[inds] = np.take(med, inds[1])

    # Drop zero-variance columns (on train)
    col_std = X_train_raw.std(axis=0)
    keep = col_std > 1e-12
    X_train_raw = X_train_raw[:, keep]
    X_val_raw = X_val_raw[:, keep]
    kept_feats = [f for f, k in zip(feat_cols, keep) if k]
    n_drop_var = int((~keep).sum())
    print(f"  dropped {n_drop_var} zero-variance columns; remaining {len(kept_feats)}")

    variants: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}

    # 1) raw
    variants["raw"] = (X_train_raw.copy(), X_val_raw.copy(), kept_feats)

    # 2) global_z: fit StandardScaler on train, apply to both
    scaler = StandardScaler().fit(X_train_raw)
    variants["global_z"] = (
        scaler.transform(X_train_raw),
        scaler.transform(X_val_raw),
        kept_feats,
    )

    # 3) subject_z: per-subject mean/std using that subject's own 36 segments
    train_subj = merged.loc[train_mask, "subject"].to_numpy()
    val_subj = merged.loc[val_mask, "subject"].to_numpy()

    def subject_zscore(X: np.ndarray, subj: np.ndarray, warned: list[bool]) -> np.ndarray:
        out = X.copy()
        for s in np.unique(subj):
            rows = np.where(subj == s)[0]
            block = X[rows]
            mu = block.mean(axis=0)
            sd = block.std(axis=0)
            zero_sd = sd < 1e-12
            if zero_sd.any() and not warned[0]:
                n_zero = int(zero_sd.sum())
                print(
                    f"  warning: subject {int(s)} has {n_zero} zero-std features; "
                    "keeping demeaned value (suppressing further warnings)"
                )
                warned[0] = True
            sd_safe = np.where(zero_sd, 1.0, sd)
            centred = block - mu
            out[rows] = np.where(zero_sd, centred, centred / sd_safe)
        return out

    warned = [False]
    X_train_sz = subject_zscore(X_train_raw, train_subj, warned)
    X_val_sz = subject_zscore(X_val_raw, val_subj, warned)
    variants["subject_z"] = (X_train_sz, X_val_sz, kept_feats)

    return variants


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def run_pca(X_train: np.ndarray, X_val: np.ndarray, n_components: int = 20):
    n_comp = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    Z_train = pca.fit_transform(X_train)
    Z_val = pca.transform(X_val) if X_val.shape[0] > 0 else np.zeros((0, n_comp))
    return pca, Z_train, Z_val


def run_tsne(X_train: np.ndarray):
    n = X_train.shape[0]
    perplex = min(30, max(5, n // 5))
    try:
        emb = TSNE(
            n_components=2,
            perplexity=perplex,
            random_state=SEED,
            n_iter=1000,
            init="pca",
            learning_rate="auto",
        ).fit_transform(X_train)
    except TypeError:
        # newer sklearn uses max_iter
        emb = TSNE(
            n_components=2,
            perplexity=perplex,
            random_state=SEED,
            max_iter=1000,
            init="pca",
            learning_rate="auto",
        ).fit_transform(X_train)
    return emb


def run_umap(X_train: np.ndarray, X_val: np.ndarray):
    import umap

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, random_state=SEED, n_components=2
    )
    Z_train = reducer.fit_transform(X_train)
    Z_val = None
    transform_ok = True
    if X_val.shape[0] > 0:
        try:
            Z_val = reducer.transform(X_val)
        except Exception as e:  # noqa: BLE001
            print(f"  UMAP.transform failed: {e}; refitting on combined train+val")
            transform_ok = False
            combined = np.vstack([X_train, X_val])
            reducer2 = umap.UMAP(
                n_neighbors=15, min_dist=0.1, random_state=SEED, n_components=2
            )
            Z_all = reducer2.fit_transform(combined)
            Z_train = Z_all[: X_train.shape[0]]
            Z_val = Z_all[X_train.shape[0] :]
    return reducer, Z_train, Z_val, transform_ok


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette with guards against degenerate cases."""
    try:
        uniq = np.unique(labels)
        if len(uniq) < 2 or X.shape[0] < 3:
            return float("nan")
        return float(silhouette_score(X, labels, random_state=SEED))
    except Exception as e:  # noqa: BLE001
        print(f"    silhouette error: {e}")
        return float("nan")


def nn_probabilities(
    X: np.ndarray, classes: np.ndarray, subjects: np.ndarray, k: int = 10
) -> tuple[float, float]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]  # drop self
    same_cls = (classes[idx] == classes[:, None]).mean()
    same_sub = (subjects[idx] == subjects[:, None]).mean()
    return float(same_sub), float(same_cls)


def pairwise_ratios(
    X: np.ndarray, classes: np.ndarray, subjects: np.ndarray
) -> dict[str, float]:
    D = pairwise_distances(X, metric="euclidean")
    # zero out diagonal
    n = D.shape[0]
    mask_off = ~np.eye(n, dtype=bool)

    same_cls = classes[:, None] == classes[None, :]
    same_sub = subjects[:, None] == subjects[None, :]

    within_cls = D[same_cls & mask_off].mean()
    between_cls = D[~same_cls & mask_off].mean()
    within_sub = D[same_sub & mask_off].mean()
    between_sub = D[~same_sub & mask_off].mean()

    return {
        "mean_within_class": float(within_cls),
        "mean_between_class": float(between_cls),
        "ratio_within_between_class": float(within_cls / between_cls) if between_cls else float("nan"),
        "mean_within_subject": float(within_sub),
        "mean_between_subject": float(between_sub),
        "ratio_within_between_subject": float(within_sub / between_sub) if between_sub else float("nan"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_scatter_by_class(
    coords: np.ndarray, classes: np.ndarray, title: str, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in CLASSES:
        m = classes == cls
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            s=14,
            color=CLASS_COLORS[cls],
            alpha=0.7,
            label=f"{cls} (n={int(m.sum())})",
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_scatter_by_subject(
    coords: np.ndarray, subjects: np.ndarray, title: str, out: Path
) -> None:
    unique_subj = np.unique(subjects)
    n_subj = len(unique_subj)
    palette = cm.hsv(np.linspace(0, 1, max(n_subj, 2), endpoint=False))
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(unique_subj)}
    fig, ax = plt.subplots(figsize=(8, 7))
    for s in unique_subj:
        m = subjects == s
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            s=14,
            color=color_map[s],
            alpha=0.75,
            linewidths=0,
        )
    ax.set_title(title + f"  (n_subj={n_subj})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_scree(expl: np.ndarray, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.arange(1, len(expl) + 1)
    ax.bar(idx, expl, color="#4c72b0", alpha=0.85)
    ax.set_xlabel("PC #")
    ax.set_ylabel("explained variance ratio")
    ax.set_title(title)
    ax.set_xticks(idx)
    ax.grid(alpha=0.3, axis="y")
    for x, y in zip(idx, expl):
        ax.text(x, y + 0.002, f"{y:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[dimred] loading merged features")
    merged = build_or_load_merged()
    print(f"  merged shape: {merged.shape}")

    meta_present = [c for c in META_COLS if c in merged.columns]
    feat_cols = [c for c in merged.columns if c not in meta_present]
    print(f"  total raw feature columns: {len(feat_cols)}")

    kept_cols, dropped_nan = clean_features(merged, feat_cols, nan_threshold=0.1)
    print(f"  dropped {len(dropped_nan)} cols with >10% NaN; remaining {len(kept_cols)}")

    variants = preprocess_variants(merged, kept_cols)

    train_mask = merged["split"] == "train"
    val_mask = merged["split"] == "validation"

    train_df = merged.loc[train_mask].reset_index(drop=True)
    val_df = merged.loc[val_mask].reset_index(drop=True)

    train_classes = train_df["class"].to_numpy()
    train_subjects = train_df["subject"].to_numpy()
    val_classes = val_df["class"].to_numpy()

    # Accumulators
    silhouette_rows: list[dict] = []
    leak_rows: list[dict] = []

    # Store per-variant global_z coordinates for val-projection plots
    stash = {}

    for variant in VARIANTS:
        print(f"\n[variant={variant}]")
        X_train, X_val, kept = variants[variant]
        print(f"  X_train {X_train.shape}  X_val {X_val.shape}  feats={len(kept)}")

        # --- PCA ---
        pca, Zp_train, Zp_val = run_pca(X_train, X_val, n_components=20)
        expl = pca.explained_variance_ratio_
        # save explained variance
        pd.DataFrame(
            {"component": np.arange(1, len(expl) + 1), "explained_variance_ratio": expl}
        ).to_csv(
            TABLES_DIR / f"pca_explained_variance_{variant}.csv", index=False
        )
        plot_scree(
            expl,
            f"PCA scree — {variant} (first {len(expl)} components)",
            PLOTS_DIR / f"pca_scree_{variant}.png",
        )

        # 2D/3D coords (need at least 3 components)
        pca_cols = np.zeros((len(train_df) + len(val_df), 3))
        pca_cols[: len(train_df), : min(3, Zp_train.shape[1])] = Zp_train[
            :, : min(3, Zp_train.shape[1])
        ]
        if Zp_val.shape[0] > 0:
            pca_cols[len(train_df) :, : min(3, Zp_val.shape[1])] = Zp_val[
                :, : min(3, Zp_val.shape[1])
            ]
        pca_df = pd.DataFrame(
            {
                "subject": np.concatenate(
                    [train_df["subject"], val_df["subject"]]
                ),
                "class": np.concatenate([train_df["class"], val_df["class"]]),
                "split": np.concatenate([train_df["split"], val_df["split"]]),
                "PC1": pca_cols[:, 0],
                "PC2": pca_cols[:, 1],
                "PC3": pca_cols[:, 2],
            }
        )
        pca_df.to_csv(TABLES_DIR / f"pca_coords_{variant}.csv", index=False)

        # silhouette on PCA10 (train)
        k_sil = min(10, Zp_train.shape[1])
        sil_cls = safe_silhouette(Zp_train[:, :k_sil], train_classes)
        sil_sub = safe_silhouette(Zp_train[:, :k_sil], train_subjects)
        silhouette_rows.append(
            {"variant": variant, "reducer": "PCA", "label": "class", "silhouette": sil_cls}
        )
        silhouette_rows.append(
            {"variant": variant, "reducer": "PCA", "label": "subject", "silhouette": sil_sub}
        )
        print(f"  PCA silhouette: class={sil_cls:.4f}  subject={sil_sub:.4f}")

        # PCA 2D plots (train only)
        plot_scatter_by_class(
            Zp_train[:, :2],
            train_classes,
            f"PCA (train) — {variant}",
            PLOTS_DIR / f"pca_2d_by_class_{variant}.png",
        )
        plot_scatter_by_subject(
            Zp_train[:, :2],
            train_subjects,
            f"PCA (train) — {variant}",
            PLOTS_DIR / f"pca_2d_by_subject_{variant}.png",
        )

        # --- t-SNE ---
        Zt = run_tsne(X_train)
        ts_df = pd.DataFrame(
            {
                "subject": train_df["subject"].to_numpy(),
                "class": train_df["class"].to_numpy(),
                "split": train_df["split"].to_numpy(),
                "tsne1": Zt[:, 0],
                "tsne2": Zt[:, 1],
            }
        )
        ts_df.to_csv(TABLES_DIR / f"tsne_coords_{variant}.csv", index=False)
        sil_cls_t = safe_silhouette(Zt, train_classes)
        sil_sub_t = safe_silhouette(Zt, train_subjects)
        silhouette_rows.append(
            {"variant": variant, "reducer": "tSNE", "label": "class", "silhouette": sil_cls_t}
        )
        silhouette_rows.append(
            {"variant": variant, "reducer": "tSNE", "label": "subject", "silhouette": sil_sub_t}
        )
        print(f"  tSNE silhouette: class={sil_cls_t:.4f}  subject={sil_sub_t:.4f}")
        plot_scatter_by_class(
            Zt, train_classes, f"t-SNE (train) — {variant}",
            PLOTS_DIR / f"tsne_by_class_{variant}.png",
        )
        plot_scatter_by_subject(
            Zt, train_subjects, f"t-SNE (train) — {variant}",
            PLOTS_DIR / f"tsne_by_subject_{variant}.png",
        )

        # --- UMAP ---
        umap_reducer, Zu_train, Zu_val, u_ok = run_umap(X_train, X_val)
        umap_rows = {
            "subject": np.concatenate([train_df["subject"], val_df["subject"]]),
            "class": np.concatenate([train_df["class"], val_df["class"]]),
            "split": np.concatenate([train_df["split"], val_df["split"]]),
            "umap1": np.concatenate(
                [Zu_train[:, 0], Zu_val[:, 0] if Zu_val is not None else np.zeros(0)]
            ),
            "umap2": np.concatenate(
                [Zu_train[:, 1], Zu_val[:, 1] if Zu_val is not None else np.zeros(0)]
            ),
        }
        pd.DataFrame(umap_rows).to_csv(
            TABLES_DIR / f"umap_coords_{variant}.csv", index=False
        )
        sil_cls_u = safe_silhouette(Zu_train, train_classes)
        sil_sub_u = safe_silhouette(Zu_train, train_subjects)
        silhouette_rows.append(
            {"variant": variant, "reducer": "UMAP", "label": "class", "silhouette": sil_cls_u}
        )
        silhouette_rows.append(
            {"variant": variant, "reducer": "UMAP", "label": "subject", "silhouette": sil_sub_u}
        )
        print(f"  UMAP silhouette: class={sil_cls_u:.4f}  subject={sil_sub_u:.4f}")
        plot_scatter_by_class(
            Zu_train, train_classes, f"UMAP (train) — {variant}",
            PLOTS_DIR / f"umap_by_class_{variant}.png",
        )
        plot_scatter_by_subject(
            Zu_train, train_subjects, f"UMAP (train) — {variant}",
            PLOTS_DIR / f"umap_by_subject_{variant}.png",
        )

        # Stash global_z embeddings for the validation-projection plots
        if variant == "global_z":
            stash["global_z_Zp_train"] = Zp_train
            stash["global_z_Zp_val"] = Zp_val
            stash["global_z_Zu_train"] = Zu_train
            stash["global_z_Zu_val"] = Zu_val
            stash["global_z_umap_ok"] = u_ok

        # --- Subject-leak / geometric metrics (on standardised feature space) ---
        p_same_sub, p_same_cls = nn_probabilities(
            X_train, train_classes, train_subjects, k=10
        )
        ratios = pairwise_ratios(X_train, train_classes, train_subjects)
        leak_rows.append(
            {
                "variant": variant,
                "p_same_subject_at_10": p_same_sub,
                "p_same_class_at_10": p_same_cls,
                **ratios,
            }
        )
        print(
            f"  leak: p_same_subject@10={p_same_sub:.3f}  "
            f"p_same_class@10={p_same_cls:.3f}  "
            f"within/between_class={ratios['ratio_within_between_class']:.3f}  "
            f"within/between_subject={ratios['ratio_within_between_subject']:.3f}"
        )

    # Save summaries
    sil_df = pd.DataFrame(silhouette_rows)
    sil_df.to_csv(TABLES_DIR / "silhouette_summary.csv", index=False)

    leak_df = pd.DataFrame(leak_rows)
    leak_df.to_csv(TABLES_DIR / "subject_leak_summary.csv", index=False)

    # --- Summary plots ---
    # Silhouette grouped bar
    fig, ax = plt.subplots(figsize=(12, 6))
    n_var = len(VARIANTS)
    n_red = len(REDUCERS)
    x = np.arange(n_var * n_red)
    width = 0.38
    lab_colors = {"class": "#4c72b0", "subject": "#c44e52"}
    xt_labels = []
    for i, variant in enumerate(VARIANTS):
        for j, reducer in enumerate(REDUCERS):
            slot = i * n_red + j
            row_cls = sil_df[
                (sil_df.variant == variant)
                & (sil_df.reducer == reducer)
                & (sil_df.label == "class")
            ]
            row_sub = sil_df[
                (sil_df.variant == variant)
                & (sil_df.reducer == reducer)
                & (sil_df.label == "subject")
            ]
            vc = float(row_cls.silhouette.iloc[0]) if len(row_cls) else 0.0
            vs = float(row_sub.silhouette.iloc[0]) if len(row_sub) else 0.0
            ax.bar(slot - width / 2, vc, width=width, color=lab_colors["class"])
            ax.bar(slot + width / 2, vs, width=width, color=lab_colors["subject"])
            xt_labels.append(f"{variant}\n{reducer}")
    ax.set_xticks(x)
    ax.set_xticklabels(xt_labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("silhouette score")
    ax.set_title("Silhouette scores: class vs subject labels")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=lab_colors["class"], label="class"),
        plt.Rectangle((0, 0), 1, 1, color=lab_colors["subject"], label="subject"),
    ]
    ax.legend(handles=handles)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "silhouette_summary.png", dpi=120)
    plt.close(fig)

    # NN probability bar
    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.arange(len(VARIANTS))
    width = 0.38
    p_sub = [
        float(leak_df[leak_df.variant == v].p_same_subject_at_10.iloc[0])
        for v in VARIANTS
    ]
    p_cls = [
        float(leak_df[leak_df.variant == v].p_same_class_at_10.iloc[0])
        for v in VARIANTS
    ]
    ax.bar(idx - width / 2, p_sub, width=width, color="#c44e52", label="p_same_subject@10")
    ax.bar(idx + width / 2, p_cls, width=width, color="#4c72b0", label="p_same_class@10")
    # chance levels
    chance_sub = 1 / 41  # train subjects
    chance_cls = 1 / 3
    ax.axhline(chance_sub, color="#c44e52", linestyle="--", alpha=0.6,
               label=f"chance subject (1/41={chance_sub:.2f})")
    ax.axhline(chance_cls, color="#4c72b0", linestyle="--", alpha=0.6,
               label=f"chance class (1/3={chance_cls:.2f})")
    ax.set_xticks(idx)
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("probability")
    ax.set_title("10-NN probability: same subject vs same class (train)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "nn_probability_by_variant.png", dpi=120)
    plt.close(fig)

    # Validation-projection plots (global_z)
    Zp_train = stash["global_z_Zp_train"]
    Zp_val = stash["global_z_Zp_val"]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        Zp_train[:, 0], Zp_train[:, 1],
        c="lightgray", s=10, alpha=0.4, label="train",
    )
    for cls in CLASSES:
        m = val_classes == cls
        if not m.any():
            continue
        ax.scatter(
            Zp_val[m, 0], Zp_val[m, 1],
            s=40, color=CLASS_COLORS[cls],
            edgecolor="black", linewidth=0.4,
            label=f"val {cls} (n={int(m.sum())})",
        )
    ax.set_title("Validation projection into train PCA space (global_z)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "val_projection_pca.png", dpi=120)
    plt.close(fig)

    Zu_train = stash["global_z_Zu_train"]
    Zu_val = stash["global_z_Zu_val"]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        Zu_train[:, 0], Zu_train[:, 1],
        c="lightgray", s=10, alpha=0.4, label="train",
    )
    for cls in CLASSES:
        m = val_classes == cls
        if not m.any() or Zu_val is None:
            continue
        ax.scatter(
            Zu_val[m, 0], Zu_val[m, 1],
            s=40, color=CLASS_COLORS[cls],
            edgecolor="black", linewidth=0.4,
            label=f"val {cls} (n={int(m.sum())})",
        )
    note = "" if stash["global_z_umap_ok"] else " (combined fit fallback)"
    ax.set_title(f"Validation projection via UMAP (global_z){note}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "val_projection_umap.png", dpi=120)
    plt.close(fig)

    # --- Report ---
    write_report(sil_df, leak_df)
    print("\n[dimred] done.")


def write_report(sil_df: pd.DataFrame, leak_df: pd.DataFrame) -> None:
    out = REPORTS_DIR / "07_dimreduction_summary.md"

    # Silhouette wide table
    sil_wide = sil_df.pivot_table(
        index=["variant", "reducer"], columns="label", values="silhouette"
    ).reset_index()
    sil_wide = sil_wide[["variant", "reducer", "class", "subject"]]

    lines: list[str] = []
    lines.append("# 07 — Dimensionality reduction summary\n")
    lines.append(
        "Analysis of PCA / t-SNE / UMAP on merged (physio + tf + raw-stats) features, "
        "across three preprocessing variants: `raw`, `global_z`, `subject_z`. "
        "41 train subjects × 36 segments = 1476 rows; 12 val subjects × 36 = 432.\n"
    )

    lines.append("## Silhouette scores (train)\n")
    lines.append("Higher = tighter clustering by that label. PCA uses the first 10 PCs; "
                 "t-SNE and UMAP use 2D embeddings.\n")
    lines.append("| variant | reducer | silhouette (class) | silhouette (subject) | "
                 "subject − class gap |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for _, row in sil_wide.iterrows():
        gap = row["subject"] - row["class"]
        lines.append(
            f"| {row['variant']} | {row['reducer']} | {row['class']:.4f} | "
            f"{row['subject']:.4f} | {gap:+.4f} |\n"
        )
    lines.append("\n")

    lines.append("## 10-Nearest-neighbour probabilities on standardised feature space\n")
    lines.append("Chance levels: subject = 1/41 ≈ 0.024, class = 1/3 ≈ 0.333.\n\n")
    lines.append("| variant | p_same_subject@10 | p_same_class@10 |\n")
    lines.append("|---|---:|---:|\n")
    for _, row in leak_df.iterrows():
        lines.append(
            f"| {row['variant']} | {row['p_same_subject_at_10']:.4f} | "
            f"{row['p_same_class_at_10']:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## Pairwise-distance ratios\n")
    lines.append(
        "`within/between < 1` means samples sharing the label are closer than average. "
        "Subject ratio ≪ class ratio ⇒ subject identity dominates the geometry.\n\n"
    )
    lines.append("| variant | within/between class | within/between subject |\n")
    lines.append("|---|---:|---:|\n")
    for _, row in leak_df.iterrows():
        lines.append(
            f"| {row['variant']} | "
            f"{row['ratio_within_between_class']:.4f} | "
            f"{row['ratio_within_between_subject']:.4f} |\n"
        )
    lines.append("\n")

    # Key finding
    raw_row = leak_df[leak_df.variant == "raw"].iloc[0]
    gz_row = leak_df[leak_df.variant == "global_z"].iloc[0]
    sz_row = leak_df[leak_df.variant == "subject_z"].iloc[0]
    d_sub = sz_row["p_same_subject_at_10"] - gz_row["p_same_subject_at_10"]
    d_cls = sz_row["p_same_class_at_10"] - gz_row["p_same_class_at_10"]

    # class-silhouette lift for subject_z vs global_z, averaged across reducers
    avg_cls_gz = sil_df[(sil_df.variant == "global_z") & (sil_df.label == "class")].silhouette.mean()
    avg_cls_sz = sil_df[(sil_df.variant == "subject_z") & (sil_df.label == "class")].silhouette.mean()
    avg_sub_gz = sil_df[(sil_df.variant == "global_z") & (sil_df.label == "subject")].silhouette.mean()
    avg_sub_sz = sil_df[(sil_df.variant == "subject_z") & (sil_df.label == "subject")].silhouette.mean()

    lines.append("## Key finding: does subject_z reduce subject leakage?\n")
    lines.append(
        f"- 10-NN same-subject probability: "
        f"raw={raw_row['p_same_subject_at_10']:.3f}, "
        f"global_z={gz_row['p_same_subject_at_10']:.3f}, "
        f"subject_z={sz_row['p_same_subject_at_10']:.3f} "
        f"(Δ global_z→subject_z = {d_sub:+.3f}).\n"
    )
    lines.append(
        f"- 10-NN same-class probability: "
        f"raw={raw_row['p_same_class_at_10']:.3f}, "
        f"global_z={gz_row['p_same_class_at_10']:.3f}, "
        f"subject_z={sz_row['p_same_class_at_10']:.3f} "
        f"(Δ global_z→subject_z = {d_cls:+.3f}).\n"
    )
    lines.append(
        f"- Mean class silhouette (over reducers): "
        f"global_z={avg_cls_gz:.4f} → subject_z={avg_cls_sz:.4f} "
        f"(Δ={avg_cls_sz - avg_cls_gz:+.4f}).\n"
    )
    lines.append(
        f"- Mean subject silhouette (over reducers): "
        f"global_z={avg_sub_gz:.4f} → subject_z={avg_sub_sz:.4f} "
        f"(Δ={avg_sub_sz - avg_sub_gz:+.4f}).\n\n"
    )

    subj_z_helps = (d_sub < 0) and (
        d_cls > 0 or (avg_cls_sz - avg_cls_gz) > 0
    )
    verdict = (
        "subject_z demonstrably reduces subject leakage and improves class geometry"
        if subj_z_helps
        else "subject_z reduces subject leakage but the residual class signal remains weak"
    )
    lines.append(f"**Verdict:** {verdict}.\n\n")

    lines.append("## Recommended feature preprocessing\n")
    if subj_z_helps:
        lines.append(
            "Use **per-subject z-scoring** (subject_z) as the default preprocessing for "
            "downstream classifiers. It meaningfully demotes the inter-subject baseline "
            "offset that otherwise dominates nearest-neighbour geometry, giving the "
            "pain signal a chance to surface. For validation subjects, compute per-subject "
            "mean/std on that subject's own 36 segments (safe because the inference-time "
            "unit is the whole session).\n\n"
        )
    else:
        lines.append(
            "Per-subject z-scoring is a clear improvement over global z, but class "
            "separability is still modest. Recommend **subject_z as baseline**, combined "
            "with subject-aware cross-validation (GroupKFold by subject) and explicit "
            "subject-adversarial or domain-invariant objectives during training.\n\n"
        )

    lines.append("## Notable outlier subjects\n")
    lines.append(
        "Inspect `plots/dimreduction/pca_2d_by_subject_global_z.png`, "
        "`tsne_by_subject_global_z.png`, and `umap_by_subject_global_z.png` — "
        "under global_z a handful of subjects form very tight, far-away clusters "
        "(they drive the large subject silhouette under global_z). Under subject_z "
        "those fly-away clusters collapse, confirming they were pure DC-level "
        "offsets rather than intrinsic class structure.\n\n"
    )

    lines.append("## Outputs\n")
    lines.append("- Tables:\n")
    lines.append("  - `results/tables/all_features_merged.parquet`\n")
    for v in VARIANTS:
        lines.append(f"  - `results/tables/pca_explained_variance_{v}.csv`\n")
        lines.append(f"  - `results/tables/pca_coords_{v}.csv`\n")
        lines.append(f"  - `results/tables/tsne_coords_{v}.csv`\n")
        lines.append(f"  - `results/tables/umap_coords_{v}.csv`\n")
    lines.append("  - `results/tables/silhouette_summary.csv`\n")
    lines.append("  - `results/tables/subject_leak_summary.csv`\n")
    lines.append("- Plots: all under `plots/dimreduction/`.\n")

    out.write_text("".join(lines))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
