"""Visualization suite for AI4Pain 2026 dataset.

Produces a thorough set of plots saved to plots/viz/.

Run:
    uv run python scripts/05_visualizations.py
from the analysis/ directory.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import CLASSES, SFREQ, SIGNALS, load_split  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["figure.dpi"] = 120
sns.set_context("talk", font_scale=0.7)

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_ROOT / "results" / "tables"
PLOTS_DIR = ANALYSIS_ROOT / "plots" / "viz"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

CLASS_COLORS = {
    "NoPain": "#4c72b0",
    "PainArm": "#dd8452",
    "PainHand": "#c44e52",
}

SIGNAL_UNITS = {
    "Bvp": "a.u.",
    "Eda": "µS",
    "Resp": "a.u.",
    "SpO2": "%",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strip_nans(x: np.ndarray) -> np.ndarray:
    """Trim trailing NaNs from 1D array."""
    mask = ~np.isnan(x)
    if not mask.any():
        return x[:0]
    last = np.where(mask)[0][-1] + 1
    return x[:last]


def spo2_is_constant(seg_spo2: np.ndarray, tol: float = 1e-6) -> bool:
    s = strip_nans(seg_spo2)
    return s.size < 2 or np.nanstd(s) < tol


def class_mask(meta: pd.DataFrame, cls: str) -> np.ndarray:
    return (meta["class"] == cls).to_numpy()


def time_axis(n: int) -> np.ndarray:
    return np.arange(n) / SFREQ


# ---------------------------------------------------------------------------
# Plot 1: example_segments_grid
# ---------------------------------------------------------------------------
def plot_example_segments_grid(train_t: np.ndarray, train_m: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=True)
    for r, cls in enumerate(CLASSES):
        cand = train_m[train_m["class"] == cls].reset_index()  # 'index' = row in tensor
        # Pick 4 representative rows (one per signal column) - but we want ONE segment
        # per (class, signal). Draw 4 distinct random picks.
        if len(cand) == 0:
            continue
        picks = RNG.choice(len(cand), size=min(4, len(cand)), replace=False)
        for c, sig in enumerate(SIGNALS):
            ax = axes[r, c]
            row = cand.iloc[picks[c % len(picks)]]
            idx = int(row["index"])
            seg = train_t[idx, SIGNALS.index(sig), :]
            seg = strip_nans(seg)
            if sig == "SpO2" and spo2_is_constant(seg):
                # fallback: find a SpO2 segment for this class that is not flat
                good = [
                    int(cand.iloc[i]["index"])
                    for i in RNG.permutation(len(cand))
                    if not spo2_is_constant(
                        train_t[int(cand.iloc[i]["index"]), SIGNALS.index("SpO2"), :]
                    )
                ]
                if good:
                    idx = good[0]
                    row = train_m.loc[idx]
                    seg = strip_nans(train_t[idx, SIGNALS.index(sig), :])
            t = time_axis(len(seg))
            ax.plot(t, seg, color=CLASS_COLORS[cls], linewidth=1.0)
            ax.set_title(
                f"{cls} — subj {int(row['subject'])} seg {int(row['segment_idx'])} "
                f"({sig})",
                fontsize=9,
            )
            if r == 2:
                ax.set_xlabel("time (s)")
            if c == 0:
                ax.set_ylabel(f"{sig} ({SIGNAL_UNITS[sig]})")
            ax.grid(alpha=0.3)
    fig.suptitle("Example segments (train): 3 classes × 4 signals", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: example_subject_3classes
# ---------------------------------------------------------------------------
def pick_good_spo2_subject(train_t: np.ndarray, train_m: pd.DataFrame) -> int:
    """Pick subject with fewest flat-SpO2 segments."""
    spo2 = train_t[:, SIGNALS.index("SpO2"), :]
    flat = np.array([spo2_is_constant(s) for s in spo2])
    by_subj = train_m.assign(_flat=flat).groupby("subject")["_flat"].mean()
    # healthy SpO2 subjects often have some variation
    candidates = by_subj[by_subj < 0.3].index.tolist()
    if not candidates:
        candidates = by_subj.sort_values().index.tolist()
    # prefer subject 5 if present (has min std > 0.4 per inventory)
    return int(5) if 5 in candidates else int(candidates[0])


def plot_example_subject_3classes(train_t: np.ndarray, train_m: pd.DataFrame, out: Path):
    sid = pick_good_spo2_subject(train_t, train_m)
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    for r, cls in enumerate(CLASSES):
        rows = train_m[(train_m["subject"] == sid) & (train_m["class"] == cls)]
        idxs = rows.index.to_numpy()
        for c, sig in enumerate(SIGNALS):
            ax = axes[r, c]
            s_i = SIGNALS.index(sig)
            for i in idxs:
                seg = strip_nans(train_t[i, s_i, :])
                if sig == "SpO2" and spo2_is_constant(seg):
                    continue
                ax.plot(time_axis(len(seg)), seg, color=CLASS_COLORS[cls],
                        alpha=0.35, linewidth=0.9)
            ax.set_title(f"{cls} — {sig}", fontsize=10)
            if r == 2:
                ax.set_xlabel("time (s)")
            if c == 0:
                ax.set_ylabel(f"{sig} ({SIGNAL_UNITS[sig]})")
            ax.grid(alpha=0.3)
    fig.suptitle(f"Subject {sid} — all 36 segments (12 per class)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots 3, 4: group mean waveforms
# ---------------------------------------------------------------------------
def _zscore_within_subject(
    train_t: np.ndarray, train_m: pd.DataFrame, n_samples: int = 1000
) -> np.ndarray:
    """Z-score each (subject, signal) combo using that subject's pooled mean/std
    across all classes and segments."""
    out = np.full((train_t.shape[0], train_t.shape[1], n_samples), np.nan,
                  dtype=np.float32)
    for sid, grp in train_m.groupby("subject"):
        rows = grp.index.to_numpy()
        for s_i, sig in enumerate(SIGNALS):
            vals = train_t[rows, s_i, :n_samples]
            flat = vals.reshape(-1)
            flat = flat[~np.isnan(flat)]
            if sig == "SpO2":
                flat = flat[flat > 2]  # drop dropouts
            if flat.size < 10 or np.std(flat) < 1e-6:
                continue
            mu, sd = float(np.mean(flat)), float(np.std(flat))
            z = (vals - mu) / sd
            # mask SpO2 constant / dropout segments
            if sig == "SpO2":
                for k, r in enumerate(rows):
                    seg = vals[k]
                    s = seg[~np.isnan(seg)]
                    if s.size < 10 or np.std(s) < 1e-6 or (s < 2).any():
                        z[k] = np.nan
            out[rows, s_i, :] = z
    return out


def plot_group_mean_waveform(
    train_t: np.ndarray, train_m: pd.DataFrame, out: Path, n_samples: int = 1000
):
    z = _zscore_within_subject(train_t, train_m, n_samples)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    t = time_axis(n_samples)
    for c, sig in enumerate(SIGNALS):
        ax = axes[c]
        s_i = SIGNALS.index(sig)
        for cls in CLASSES:
            mask = class_mask(train_m, cls)
            arr = z[mask, s_i, :]  # (n_seg, 1000)
            mean = np.nanmean(arr, axis=0)
            n = np.sum(~np.isnan(arr), axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
            ax.plot(t, mean, color=CLASS_COLORS[cls], label=cls, linewidth=1.6)
            ax.fill_between(t, mean - sem, mean + sem,
                            color=CLASS_COLORS[cls], alpha=0.2)
        ax.set_title(f"{sig} (within-subject z-score)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("z")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Group mean waveform per class (train, ±1 SEM)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_group_mean_waveform_subject_norm(
    train_t: np.ndarray, train_m: pd.DataFrame, out: Path, n_samples: int = 1000
):
    """Z-score each segment against that subject's NoPain baseline stats."""
    n_seg, n_sig, _ = train_t.shape
    z = np.full((n_seg, n_sig, n_samples), np.nan, dtype=np.float32)
    for sid, grp in train_m.groupby("subject"):
        nopain_rows = grp[grp["class"] == "NoPain"].index.to_numpy()
        all_rows = grp.index.to_numpy()
        if len(nopain_rows) == 0:
            continue
        for s_i, sig in enumerate(SIGNALS):
            base = train_t[nopain_rows, s_i, :n_samples].reshape(-1)
            base = base[~np.isnan(base)]
            if sig == "SpO2":
                base = base[base > 2]
            if base.size < 10 or np.std(base) < 1e-6:
                continue
            mu, sd = float(np.mean(base)), float(np.std(base))
            vals = train_t[all_rows, s_i, :n_samples]
            zz = (vals - mu) / sd
            if sig == "SpO2":
                for k, r in enumerate(all_rows):
                    seg = vals[k]
                    s = seg[~np.isnan(seg)]
                    if s.size < 10 or np.std(s) < 1e-6 or (s < 2).any():
                        zz[k] = np.nan
            z[all_rows, s_i, :] = zz

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    t = time_axis(n_samples)
    for c, sig in enumerate(SIGNALS):
        ax = axes[c]
        s_i = SIGNALS.index(sig)
        for cls in CLASSES:
            mask = class_mask(train_m, cls)
            arr = z[mask, s_i, :]
            mean = np.nanmean(arr, axis=0)
            n = np.sum(~np.isnan(arr), axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
            ax.plot(t, mean, color=CLASS_COLORS[cls], label=cls, linewidth=1.6)
            ax.fill_between(t, mean - sem, mean + sem,
                            color=CLASS_COLORS[cls], alpha=0.2)
        ax.set_title(f"{sig} (baseline-normalized z)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("z vs NoPain baseline")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Group mean waveform per class — normalized by subject's NoPain baseline",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: spectrograms
# ---------------------------------------------------------------------------
def plot_spectrograms(train_t: np.ndarray, train_m: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    for r, cls in enumerate(CLASSES):
        cand = train_m[train_m["class"] == cls]
        for c, sig in enumerate(SIGNALS):
            ax = axes[r, c]
            s_i = SIGNALS.index(sig)
            # pick segment with decent length and not flat for SpO2
            pool = []
            for i in cand.index:
                seg = strip_nans(train_t[i, s_i, :])
                if seg.size < 256:
                    continue
                if sig == "SpO2" and spo2_is_constant(seg):
                    continue
                pool.append((i, seg))
                if len(pool) > 20:
                    break
            if not pool:
                ax.axis("off")
                ax.set_title(f"{cls} — {sig}: no valid seg", fontsize=9)
                continue
            i, seg = pool[int(RNG.integers(len(pool)))]
            f, t_, Sxx = sp_signal.spectrogram(
                seg, fs=SFREQ, nperseg=128, noverlap=64
            )
            Sxx_db = 10 * np.log10(Sxx + 1e-12)
            im = ax.pcolormesh(t_, f, Sxx_db, shading="auto", cmap="magma")
            ax.set_title(f"{cls} — {sig} (subj {int(train_m.loc[i,'subject'])})",
                         fontsize=9)
            if r == 2:
                ax.set_xlabel("time (s)")
            if c == 0:
                ax.set_ylabel("freq (Hz)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")
    fig.suptitle("Spectrogram examples (nperseg=128, overlap=64)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: violins
# ---------------------------------------------------------------------------
def plot_class_signal_violins_raw(raw_stats: pd.DataFrame, out: Path):
    df = raw_stats[raw_stats["split"] == "train"].copy()
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for c, sig in enumerate(SIGNALS):
        for r, stat in enumerate(["mean", "std"]):
            ax = axes[r, c]
            col = f"{sig}_{stat}"
            sub = df[["class", col]].dropna().copy()
            # filter SpO2 flats
            note = ""
            if sig == "SpO2":
                before = len(sub)
                sub = sub[df[f"{sig}_std"] > 1e-6]
                note = f"\n(n={len(sub)}/{before} after flat filter)"
            order = list(CLASSES)
            sns.violinplot(
                data=sub, x="class", y=col, hue="class", order=order,
                hue_order=order, ax=ax, legend=False,
                palette={k: CLASS_COLORS[k] for k in order},
                cut=0, inner="quartile",
            )
            ax.set_title(f"{sig} {stat}{note}", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel(col)
            ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Per-segment raw stats per class (train)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7: subject radar
# ---------------------------------------------------------------------------
def plot_subject_variability_radar(physio: pd.DataFrame, out: Path):
    df = physio[physio["split"] == "train"].copy()
    feat_cols = [
        "BVP_hr_mean", "EDA_scr_count", "RESP_rate",
        "EDA_tonic_mean", "BVP_peak_amp_mean", "SPO2_mean",
    ]
    # build per-(subject, class) means
    per = df.groupby(["subject", "class"])[feat_cols].mean().reset_index()
    # global z-score each feature for radar scale
    for col in feat_cols:
        mu, sd = per[col].mean(), per[col].std()
        if sd > 0:
            per[col] = (per[col] - mu) / sd
    # pick 6 random subjects that have all three classes
    subj_counts = per.groupby("subject")["class"].nunique()
    avail = subj_counts[subj_counts == 3].index.tolist()
    RNG.shuffle(avail)
    pick = sorted(avail[:6])
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), subplot_kw={"projection": "polar"})
    angles = np.linspace(0, 2 * np.pi, len(feat_cols), endpoint=False).tolist()
    angles += angles[:1]
    for ax, sid in zip(axes.flat, pick):
        for cls in CLASSES:
            row = per[(per["subject"] == sid) & (per["class"] == cls)]
            if row.empty:
                continue
            vals = row[feat_cols].iloc[0].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, color=CLASS_COLORS[cls], label=cls, linewidth=1.5)
            ax.fill(angles, vals, color=CLASS_COLORS[cls], alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feat_cols, fontsize=7)
        ax.set_title(f"Subject {sid}", fontsize=11, pad=15)
        ax.grid(alpha=0.4)
    axes.flat[0].legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    fig.suptitle("Per-subject physio feature radar (z-scored, 6 random subjects)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 8: correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap_raw_stats(raw_stats: pd.DataFrame, out: Path):
    df = raw_stats[raw_stats["split"] == "train"].copy()
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ["subject", "segment_idx"] if c in df.columns],
        errors="ignore",
    )
    # pick top 30 by variance
    variances = numeric.var().sort_values(ascending=False)
    cols = variances.head(30).index.tolist()
    corr = numeric[cols].corr()
    try:
        g = sns.clustermap(
            corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            figsize=(13, 13), xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Pearson r"},
        )
        g.fig.suptitle("Raw-stat correlation clustermap (top-30 variance, train)",
                       y=1.02, fontsize=13)
        g.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(g.fig)
    except Exception as e:
        print(f"  clustermap fallback: {e}")
        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title("Raw-stat correlation heatmap")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 9: EDA tonic slope distribution
# ---------------------------------------------------------------------------
def plot_eda_tonic_slope(physio: pd.DataFrame, out: Path):
    df = physio[physio["split"] == "train"].copy()
    col = "EDA_tonic_slope"
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in CLASSES:
        vals = df[df["class"] == cls][col].dropna().to_numpy()
        # clip extreme outliers for display
        lo, hi = np.percentile(vals, [1, 99])
        vals_plot = vals[(vals >= lo) & (vals <= hi)]
        ax.hist(vals_plot, bins=50, alpha=0.45, label=f"{cls} (n={len(vals)})",
                color=CLASS_COLORS[cls], density=True)
        m = np.nanmean(vals)
        ax.axvline(m, color=CLASS_COLORS[cls], linestyle="--", linewidth=2,
                   label=f"{cls} mean={m:.3f}")
    ax.set_xlabel("EDA tonic slope (µS/s)")
    ax.set_ylabel("density")
    ax.set_title("EDA tonic slope distribution per class (train)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 10: BVP HR KDE
# ---------------------------------------------------------------------------
def plot_hr_distribution(physio: pd.DataFrame, out: Path):
    df = physio[physio["split"] == "train"].copy()
    col = "BVP_hr_mean"
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in CLASSES:
        vals = df[df["class"] == cls][col].dropna()
        vals = vals[(vals > 30) & (vals < 200)]  # plausibility
        sns.kdeplot(vals, ax=ax, color=CLASS_COLORS[cls], linewidth=2,
                    label=f"{cls} (n={len(vals)}, μ={vals.mean():.1f})",
                    fill=True, alpha=0.15)
    ax.set_xlabel("heart rate (bpm)")
    ax.set_ylabel("density")
    ax.set_title("BVP heart-rate distribution per class (train)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 11: PSD per class per signal
# ---------------------------------------------------------------------------
def plot_psd_per_class(train_t: np.ndarray, train_m: pd.DataFrame, out_dir: Path):
    tf_path = TABLES_DIR / "tf_features.parquet"
    if not tf_path.exists():
        print("  tf_features.parquet missing; computing PSDs on the fly.")
    for s_i, sig in enumerate(SIGNALS):
        fig, ax = plt.subplots(figsize=(10, 6))
        for cls in CLASSES:
            mask = class_mask(train_m, cls)
            psds = []
            freqs = None
            for idx in np.where(mask)[0]:
                seg = strip_nans(train_t[idx, s_i, :])
                if seg.size < 256:
                    continue
                if sig == "SpO2" and (spo2_is_constant(seg) or (seg < 2).any()):
                    continue
                f, p = sp_signal.welch(seg, fs=SFREQ, nperseg=256)
                freqs = f
                psds.append(p)
            if not psds:
                continue
            psd_mean = np.mean(np.vstack(psds), axis=0)
            ax.loglog(freqs[1:], psd_mean[1:], color=CLASS_COLORS[cls],
                      label=f"{cls} (n={len(psds)})", linewidth=1.6)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"Average Welch PSD — {sig} (train)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"psd_per_class_{sig}.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 12: segment position effect
# ---------------------------------------------------------------------------
def plot_segment_position_effect(physio: pd.DataFrame, out: Path):
    df = physio[physio["split"] == "train"].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    for cls in CLASSES:
        sub = df[df["class"] == cls]
        grp = sub.groupby("segment_idx")["BVP_hr_mean"].agg(["mean", "std", "count"])
        sem = grp["std"] / np.sqrt(grp["count"])
        ax.plot(grp.index, grp["mean"], color=CLASS_COLORS[cls], marker="o",
                label=cls, linewidth=1.6)
        ax.fill_between(grp.index, grp["mean"] - sem, grp["mean"] + sem,
                        color=CLASS_COLORS[cls], alpha=0.2)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("segment index (1..12)")
    ax.set_ylabel("mean BVP heart rate (bpm)")
    ax.set_title("Segment-position effect on HR (mean ± SEM across subjects, train)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 13: t-SNE of (subject, class) mean feature vectors
# ---------------------------------------------------------------------------
def plot_per_subject_class_ordering(physio: pd.DataFrame, out: Path):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    df = physio[physio["split"] == "train"].copy()
    feat_cols = [c for c in df.columns
                 if c not in {"split", "subject", "class", "segment_idx", "segment_id"}]
    # per (subject, class) mean
    agg = df.groupby(["subject", "class"])[feat_cols].mean().reset_index()
    X = agg[feat_cols].to_numpy(dtype=float)
    # impute NaNs with column median; fall back to 0 for all-NaN columns
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    # drop columns that are still constant (would give NaN after StandardScaler)
    col_std = X.std(axis=0)
    keep = col_std > 0
    X = X[:, keep]
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    perplex = max(5, min(30, len(X) // 3))
    emb = TSNE(n_components=2, perplexity=perplex, random_state=SEED,
               init="pca", learning_rate="auto").fit_transform(X)
    agg["x"], agg["y"] = emb[:, 0], emb[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # colored by class
    ax = axes[0]
    for cls in CLASSES:
        sub = agg[agg["class"] == cls]
        ax.scatter(sub["x"], sub["y"], color=CLASS_COLORS[cls], label=cls,
                   s=55, edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.set_title("t-SNE of (subject,class) mean feature vectors — by class")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.3)
    ax.legend()

    # colored by subject to reveal subject clusters
    ax = axes[1]
    subj_colors = plt.cm.tab20(
        np.linspace(0, 1, agg["subject"].nunique())
    )
    subj_map = {s: subj_colors[i] for i, s in enumerate(sorted(agg["subject"].unique()))}
    for _, row in agg.iterrows():
        ax.scatter(row["x"], row["y"], color=subj_map[row["subject"]],
                   s=55, edgecolor="black", linewidth=0.4)
    # connect 3 class points of same subject
    for sid, grp in agg.groupby("subject"):
        pts = grp[["x", "y"]].to_numpy()
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                        color=subj_map[sid], alpha=0.25, linewidth=0.7)
    ax.set_title("Same view, coloured by subject (lines link a subject's 3 class means)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.3)

    fig.suptitle("Subject effect vs class effect — physio-feature t-SNE", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[viz] seed={SEED}, output dir={PLOTS_DIR}")
    train_t, train_m = load_split("train")
    print(f"  train tensor: {train_t.shape}, meta rows: {len(train_m)}")

    # Tables
    raw_stats_fp = TABLES_DIR / "raw_stats_per_segment.parquet"
    physio_fp = TABLES_DIR / "physio_features.parquet"
    raw_stats = pd.read_parquet(raw_stats_fp)
    physio = pd.read_parquet(physio_fp)
    print(f"  raw_stats: {raw_stats.shape}, physio: {physio.shape}")

    print("[1] example_segments_grid.png")
    plot_example_segments_grid(train_t, train_m, PLOTS_DIR / "example_segments_grid.png")

    print("[2] example_subject_3classes.png")
    plot_example_subject_3classes(train_t, train_m,
                                  PLOTS_DIR / "example_subject_3classes.png")

    print("[3] group_mean_waveform_per_class.png")
    plot_group_mean_waveform(train_t, train_m,
                             PLOTS_DIR / "group_mean_waveform_per_class.png")

    print("[4] group_mean_waveform_per_class_subject_normalized.png")
    plot_group_mean_waveform_subject_norm(
        train_t, train_m,
        PLOTS_DIR / "group_mean_waveform_per_class_subject_normalized.png",
    )

    print("[5] spectrogram_examples.png")
    plot_spectrograms(train_t, train_m, PLOTS_DIR / "spectrogram_examples.png")

    print("[6] class_signal_violins_raw.png")
    plot_class_signal_violins_raw(raw_stats, PLOTS_DIR / "class_signal_violins_raw.png")

    print("[7] subject_variability_radar.png")
    plot_subject_variability_radar(physio, PLOTS_DIR / "subject_variability_radar.png")

    print("[8] correlation_heatmap_raw_stats.png")
    plot_correlation_heatmap_raw_stats(raw_stats,
                                       PLOTS_DIR / "correlation_heatmap_raw_stats.png")

    print("[9] eda_tonic_slope_distribution.png")
    plot_eda_tonic_slope(physio, PLOTS_DIR / "eda_tonic_slope_distribution.png")

    print("[10] hr_distribution_by_class.png")
    plot_hr_distribution(physio, PLOTS_DIR / "hr_distribution_by_class.png")

    print("[11] psd_per_class_{signal}.png")
    plot_psd_per_class(train_t, train_m, PLOTS_DIR)

    print("[12] segment_position_effect.png")
    plot_segment_position_effect(physio, PLOTS_DIR / "segment_position_effect.png")

    print("[13] per_subject_class_ordering.png")
    plot_per_subject_class_ordering(physio, PLOTS_DIR / "per_subject_class_ordering.png")

    print("[viz] done.")


if __name__ == "__main__":
    main()
