"""Time and frequency domain feature extraction for the AI4Pain 2026 dataset.

Outputs
-------
- results/tables/tf_features.parquet
- results/tables/tf_features_dictionary.csv
- results/tables/tf_features_class_means.csv
- results/tables/tf_features_anova_train.csv
- plots/tfdomain/{psd_mean_per_class_*.png, band_power_bar_*.png,
                  spectral_entropy_violin.png, hjorth_scatter.png,
                  top16_tf_features_box.png}
- results/reports/04_tfdomain_features_summary.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as sps
from scipy import stats as sstats
from tqdm import tqdm

from src.data_loader import CLASSES, SFREQ, SIGNALS, load_split  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import neurokit2 as nk  # noqa: E402
    _HAS_NK = True
except Exception:
    _HAS_NK = False


# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"
REPORTS_DIR = ROOT / "results" / "reports"
PLOTS_DIR = ROOT / "plots" / "tfdomain"
for d in (TABLES_DIR, REPORTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# Physiological bands per signal (Hz)
BANDS: dict[str, dict[str, tuple[float, float]]] = {
    "Bvp": {
        "vlf": (0.003, 0.04),
        "lf": (0.04, 0.15),
        "hf": (0.15, 0.4),
        "vhf": (0.4, 4.0),
    },
    "Eda": {
        "tonic": (0.01, 0.1),
        "phasic": (0.1, 0.5),
        "higher": (0.5, 5.0),
    },
    "Resp": {
        "breathing": (0.1, 0.5),
    },
    "SpO2": {
        "low": (0.0, 0.1),
    },
}


# ---------------------------------------------------------------------------
# Helper: strip trailing NaNs
# ---------------------------------------------------------------------------
def strip_nans(x: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(x)
    if not mask.any():
        return np.array([], dtype=np.float32)
    # assume trailing NaN padding; keep leading valid stretch
    last = np.nonzero(mask)[0][-1] + 1
    return x[:last].astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Time-domain helpers
# ---------------------------------------------------------------------------
def _hjorth(x: np.ndarray) -> tuple[float, float, float]:
    if x.size < 3:
        return np.nan, np.nan, np.nan
    var0 = np.var(x)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var1 = np.var(dx) if dx.size else np.nan
    var2 = np.var(ddx) if ddx.size else np.nan
    if var0 <= 0 or var1 <= 0:
        return float(var0), np.nan, np.nan
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 else np.nan
    return float(var0), float(mobility), float(complexity)


def _sample_entropy(x: np.ndarray, m: int = 2, r_coef: float = 0.2) -> float:
    """Simple fallback SampEn (Chebyshev distance)."""
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N < m + 2:
        return np.nan
    r = r_coef * np.std(x)
    if r <= 0:
        return np.nan
    # subsample for speed on long segments
    if N > 600:
        x = x[::2]
        N = x.size

    def _phi(mm: int) -> float:
        templates = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        # broadcasting Chebyshev distance
        dist = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        # exclude self-matches
        np.fill_diagonal(dist, np.inf)
        return np.sum(dist <= r) / (N - mm + 1)

    try:
        B = _phi(m)
        A = _phi(m + 1)
        if B == 0 or A == 0:
            return np.nan
        return float(-np.log(A / B))
    except Exception:
        return np.nan


def _petrosian_fd(x: np.ndarray) -> float:
    if x.size < 3:
        return np.nan
    dx = np.diff(x)
    # sign changes
    N_delta = int(np.sum(np.diff(np.signbit(dx).astype(int)) != 0))
    N = x.size
    denom = np.log10(N) + np.log10(N / (N + 0.4 * N_delta))
    if denom <= 0:
        return np.nan
    return float(np.log10(N) / denom)


def _zero_cross_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    xc = x - np.mean(x)
    return float(np.sum(np.diff(np.signbit(xc).astype(int)) != 0) / (x.size - 1))


def _mean_cross_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    m = np.mean(x)
    above = (x > m).astype(int)
    return float(np.sum(np.abs(np.diff(above))) / (x.size - 1))


def compute_time_features(x: np.ndarray, sfreq: float) -> dict[str, float]:
    out: dict[str, float] = {}
    if x.size < 4:
        return out
    xm = np.mean(x)
    xs = np.std(x)
    out["mean"] = float(xm)
    out["std"] = float(xs)
    out["min"] = float(np.min(x))
    out["max"] = float(np.max(x))
    out["p10"] = float(np.percentile(x, 10))
    out["p90"] = float(np.percentile(x, 90))
    out["iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    out["skew"] = float(sstats.skew(x)) if xs > 0 else 0.0
    out["kurtosis"] = float(sstats.kurtosis(x)) if xs > 0 else 0.0
    out["rms"] = float(np.sqrt(np.mean(x ** 2)))
    out["mean_abs"] = float(np.mean(np.abs(x)))
    out["mad"] = float(np.median(np.abs(x - np.median(x))))
    out["energy"] = float(np.sum(x ** 2))
    out["zcr"] = _zero_cross_rate(x)
    out["mcr"] = _mean_cross_rate(x)
    act, mob, cmp_ = _hjorth(x)
    out["hjorth_activity"] = act
    out["hjorth_mobility"] = mob
    out["hjorth_complexity"] = cmp_

    # Sample entropy (nk if available, else fallback)
    try:
        if _HAS_NK:
            val, _ = nk.entropy_sample(x, dimension=2, tolerance=0.2 * xs if xs > 0 else None)
            out["samp_entropy"] = float(val) if np.isfinite(val) else np.nan
        else:
            out["samp_entropy"] = _sample_entropy(x)
    except Exception:
        out["samp_entropy"] = np.nan

    # Approximate entropy
    try:
        if _HAS_NK:
            val, _ = nk.entropy_approximate(x, dimension=2, tolerance=0.2 * xs if xs > 0 else None)
            out["approx_entropy"] = float(val) if np.isfinite(val) else np.nan
        else:
            out["approx_entropy"] = np.nan
    except Exception:
        out["approx_entropy"] = np.nan

    # Permutation entropy
    try:
        if _HAS_NK:
            val, _ = nk.entropy_permutation(x, dimension=3, delay=1)
            out["perm_entropy"] = float(val) if np.isfinite(val) else np.nan
        else:
            # manual PE, m=3
            m = 3
            if x.size >= m + 1:
                from itertools import permutations
                patterns = list(permutations(range(m)))
                counts = {p: 0 for p in patterns}
                for i in range(x.size - m + 1):
                    sub = x[i:i + m]
                    order = tuple(np.argsort(sub))
                    counts[order] = counts.get(order, 0) + 1
                total = sum(counts.values())
                p = np.array([c for c in counts.values() if c > 0]) / total
                out["perm_entropy"] = float(-np.sum(p * np.log(p)) / np.log(np.math.factorial(m)))
            else:
                out["perm_entropy"] = np.nan
    except Exception:
        out["perm_entropy"] = np.nan

    out["petrosian_fd"] = _petrosian_fd(x)

    # DFA alpha
    try:
        if _HAS_NK and x.size >= 100:
            val, _ = nk.fractal_dfa(x)
            out["dfa_alpha"] = float(val) if np.isfinite(val) else np.nan
        else:
            out["dfa_alpha"] = np.nan
    except Exception:
        out["dfa_alpha"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Frequency-domain
# ---------------------------------------------------------------------------
def _psd(x: np.ndarray, sfreq: float, nperseg: int = 256, noverlap: int = 128):
    if x.size < 16:
        return np.array([]), np.array([])
    nps = min(nperseg, x.size)
    nov = min(noverlap, nps - 1) if nps > 1 else 0
    f, pxx = sps.welch(x, fs=sfreq, nperseg=nps, noverlap=nov,
                       detrend="constant", scaling="density")
    return f, pxx


def _band_power(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    if f.size == 0:
        return np.nan
    mask = (f >= lo) & (f <= hi)
    if not mask.any():
        return 0.0
    return float(np.trapezoid(pxx[mask], f[mask]))


def _spectral_entropy(pxx: np.ndarray) -> float:
    if pxx.size == 0:
        return np.nan
    p = pxx / (pxx.sum() + 1e-12)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(-np.sum(p * np.log(p)) / np.log(p.size))


def _spectral_centroid(f: np.ndarray, pxx: np.ndarray) -> float:
    if f.size == 0 or pxx.sum() <= 0:
        return np.nan
    return float(np.sum(f * pxx) / np.sum(pxx))


def _spectral_spread(f: np.ndarray, pxx: np.ndarray, centroid: float) -> float:
    if f.size == 0 or pxx.sum() <= 0 or not np.isfinite(centroid):
        return np.nan
    return float(np.sqrt(np.sum(((f - centroid) ** 2) * pxx) / np.sum(pxx)))


def _spectral_flatness(pxx: np.ndarray) -> float:
    p = pxx[pxx > 0]
    if p.size == 0:
        return np.nan
    g = np.exp(np.mean(np.log(p)))
    a = np.mean(p)
    return float(g / a) if a > 0 else np.nan


def _spectral_edge(f: np.ndarray, pxx: np.ndarray, frac: float = 0.95) -> float:
    if f.size == 0 or pxx.sum() <= 0:
        return np.nan
    cs = np.cumsum(pxx)
    cs /= cs[-1]
    idx = np.searchsorted(cs, frac)
    idx = min(idx, f.size - 1)
    return float(f[idx])


def _spectral_bandwidth(f: np.ndarray, pxx: np.ndarray) -> float:
    # root-mean-square bandwidth around centroid (== spectral spread)
    return _spectral_spread(f, pxx, _spectral_centroid(f, pxx))


def compute_freq_features(x: np.ndarray, sfreq: float, signal_name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    f, pxx = _psd(x, sfreq)
    if f.size == 0:
        return out

    total_power = float(np.trapezoid(pxx, f))
    out["total_power"] = total_power
    out["spec_entropy"] = _spectral_entropy(pxx)
    centroid = _spectral_centroid(f, pxx)
    out["spec_centroid"] = centroid
    out["spec_spread"] = _spectral_spread(f, pxx, centroid)
    out["spec_bandwidth"] = out["spec_spread"]
    out["spec_flatness"] = _spectral_flatness(pxx)
    out["spec_edge_95"] = _spectral_edge(f, pxx, 0.95)

    # mean frequency (power-weighted mean)
    out["mean_freq"] = centroid

    # dominant / peak frequency
    if pxx.size:
        peak_idx = int(np.argmax(pxx))
        out["peak_freq"] = float(f[peak_idx])
        out["peak_power"] = float(pxx[peak_idx])
        out["peak_rel_power"] = float(pxx[peak_idx] / (pxx.sum() + 1e-12))
    else:
        out["peak_freq"] = np.nan
        out["peak_power"] = np.nan
        out["peak_rel_power"] = np.nan

    # Band powers (physiological)
    bands = BANDS.get(signal_name, {})
    for name, (lo, hi) in bands.items():
        bp = _band_power(f, pxx, lo, hi)
        out[f"bp_{name}"] = bp
        out[f"bp_{name}_rel"] = bp / total_power if total_power > 0 else np.nan

    # Dominant freq within the most physiologically-meaningful band
    if signal_name == "Bvp":
        # LF/HF ratio
        lf = out.get("bp_lf", np.nan)
        hf = out.get("bp_hf", np.nan)
        out["lf_hf_ratio"] = float(lf / hf) if (hf and hf > 0 and np.isfinite(hf)) else np.nan
        # dom freq restricted to HF (~cardiac)
        lo, hi = BANDS["Bvp"]["vhf"]
        m = (f >= lo) & (f <= hi)
        if m.any():
            out["dom_freq_cardiac"] = float(f[m][int(np.argmax(pxx[m]))])
        else:
            out["dom_freq_cardiac"] = np.nan
    elif signal_name == "Resp":
        lo, hi = BANDS["Resp"]["breathing"]
        m = (f >= lo) & (f <= hi)
        if m.any():
            out["dom_freq_breathing"] = float(f[m][int(np.argmax(pxx[m]))])
        else:
            out["dom_freq_breathing"] = np.nan
    elif signal_name == "SpO2":
        out["total_variance"] = float(np.var(x)) if x.size else np.nan

    return out


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
TIME_CATEGORY = {
    "mean": "time", "std": "time", "min": "time", "max": "time",
    "p10": "time", "p90": "time", "iqr": "time", "skew": "time",
    "kurtosis": "time", "rms": "time", "mean_abs": "time", "mad": "time",
    "energy": "time", "zcr": "time", "mcr": "time",
    "hjorth_activity": "time", "hjorth_mobility": "time",
    "hjorth_complexity": "time",
    "samp_entropy": "entropy", "approx_entropy": "entropy",
    "perm_entropy": "entropy",
    "petrosian_fd": "fractal", "dfa_alpha": "fractal",
}

FREQ_FEATURES_DESC = {
    "total_power": "Integrated PSD across all freqs",
    "spec_entropy": "Shannon entropy of normalised PSD",
    "spec_centroid": "Power-weighted mean frequency",
    "spec_spread": "Power-weighted std around centroid",
    "spec_bandwidth": "RMS bandwidth (same as spread)",
    "spec_flatness": "Geometric/arithmetic mean ratio of PSD",
    "spec_edge_95": "Frequency below which 95% of power lies",
    "mean_freq": "Mean frequency (=centroid)",
    "peak_freq": "Frequency of PSD maximum",
    "peak_power": "PSD at peak frequency",
    "peak_rel_power": "Peak power / total power",
    "lf_hf_ratio": "LF / HF band power ratio (BVP)",
    "dom_freq_cardiac": "Dominant freq in BVP 0.4-4 Hz",
    "dom_freq_breathing": "Dominant freq in RESP 0.1-0.5 Hz",
    "total_variance": "Time-domain variance (SpO2)",
}

TIME_FEATURES_DESC = {
    "mean": "Mean",
    "std": "Standard deviation",
    "min": "Minimum",
    "max": "Maximum",
    "p10": "10th percentile",
    "p90": "90th percentile",
    "iqr": "Interquartile range",
    "skew": "Fisher skewness",
    "kurtosis": "Excess kurtosis",
    "rms": "Root mean square",
    "mean_abs": "Mean absolute value",
    "mad": "Median absolute deviation",
    "energy": "Sum of squared amplitudes",
    "zcr": "Zero-crossing rate (mean-subtracted)",
    "mcr": "Mean-crossing rate",
    "hjorth_activity": "Variance",
    "hjorth_mobility": "Sqrt(var(dx)/var(x))",
    "hjorth_complexity": "Mobility(dx)/Mobility(x)",
    "samp_entropy": "Sample entropy (m=2, r=0.2*std)",
    "approx_entropy": "Approximate entropy (m=2)",
    "perm_entropy": "Permutation entropy (m=3)",
    "petrosian_fd": "Petrosian fractal dimension",
    "dfa_alpha": "Detrended fluctuation analysis alpha",
}


def compute_all(x: np.ndarray, sfreq: float, signal_name: str) -> dict[str, float]:
    x = strip_nans(x)
    feats = {}
    if x.size < 4:
        return feats
    t = compute_time_features(x, sfreq)
    f = compute_freq_features(x, sfreq, signal_name)
    for k, v in t.items():
        feats[f"{signal_name}_{k}"] = v
    for k, v in f.items():
        feats[f"{signal_name}_{k}"] = v
    return feats


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------
def process_split(split: str) -> tuple[pd.DataFrame, dict]:
    tensor, meta = load_split(split)
    rows: list[dict] = []
    fail_counts: dict[str, int] = {}
    for i in tqdm(range(len(meta)), desc=f"features {split}"):
        base = {
            "split": split,
            "subject": int(meta.iloc[i]["subject"]),
            "class": meta.iloc[i]["class"],
            "segment_idx": int(meta.iloc[i]["segment_idx"]),
            "segment_id": meta.iloc[i]["segment_id"],
        }
        feats: dict[str, float] = {}
        for s_i, sig in enumerate(SIGNALS):
            try:
                f_dict = compute_all(tensor[i, s_i], SFREQ, sig)
                feats.update(f_dict)
            except Exception:
                fail_counts[sig] = fail_counts.get(sig, 0) + 1
        rows.append({**base, **feats})
    return pd.DataFrame(rows), fail_counts


def build_dictionary(all_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in all_cols:
        if col in {"split", "subject", "class", "segment_idx", "segment_id"}:
            continue
        sig, feat = col.split("_", 1)
        if feat in TIME_CATEGORY:
            category = TIME_CATEGORY[feat]
            desc = TIME_FEATURES_DESC.get(feat, "")
        elif feat.startswith("bp_"):
            category = "freq"
            band = feat[3:]
            if band.endswith("_rel"):
                band = band[:-4]
                desc = f"Relative band power in {band} band"
            else:
                desc = f"Band power in {band} band"
        elif feat in FREQ_FEATURES_DESC:
            category = "freq"
            desc = FREQ_FEATURES_DESC[feat]
        else:
            category = "other"
            desc = ""
        rows.append({
            "feature_name": col,
            "signal": sig,
            "category": category,
            "description": desc,
        })
    return pd.DataFrame(rows)


def class_means_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(["split", "class"])[feature_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.reset_index()


def anova_table(df_train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # subject-mean values per class
    subj_mean = df_train.groupby(["subject", "class"])[feature_cols].mean().reset_index()
    rows = []
    for feat in feature_cols:
        groups = [subj_mean.loc[subj_mean["class"] == c, feat].dropna().values
                  for c in CLASSES]
        if any(g.size < 2 for g in groups):
            rows.append({"feature": feat, "F": np.nan, "p": np.nan, "n": 0})
            continue
        try:
            F, p = sstats.f_oneway(*groups)
            rows.append({"feature": feat, "F": float(F), "p": float(p),
                         "n": int(sum(g.size for g in groups))})
        except Exception:
            rows.append({"feature": feat, "F": np.nan, "p": np.nan, "n": 0})
    return pd.DataFrame(rows).sort_values("F", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
CLASS_PALETTE = {"NoPain": "#3b82f6", "PainArm": "#f59e0b", "PainHand": "#ef4444"}


def plot_psd_mean_per_class(tensor_train: np.ndarray, meta_train: pd.DataFrame,
                             sfreq: float) -> None:
    for s_i, sig in enumerate(SIGNALS):
        # compute PSD per segment (train split)
        fvec = None
        psd_by_class: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
        for i in range(len(meta_train)):
            x = strip_nans(tensor_train[i, s_i])
            if x.size < 16:
                continue
            f, pxx = _psd(x, sfreq)
            if fvec is None:
                fvec = f
            if len(f) == len(fvec):
                psd_by_class[meta_train.iloc[i]["class"]].append(pxx)
        if fvec is None:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for c in CLASSES:
            if not psd_by_class[c]:
                continue
            P = np.vstack(psd_by_class[c])
            m = P.mean(axis=0)
            s = P.std(axis=0)
            ax.plot(fvec, m, color=CLASS_PALETTE[c], label=c, linewidth=1.5)
            ax.fill_between(fvec, np.maximum(m - s, 1e-20), m + s,
                            color=CLASS_PALETTE[c], alpha=0.15)
        if sig == "SpO2":
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        else:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"Mean PSD per class — {sig}")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"psd_mean_per_class_{sig}.png", dpi=130)
        plt.close(fig)


def plot_band_power_bars(df: pd.DataFrame) -> None:
    train = df[df["split"] == "train"]
    for sig in SIGNALS:
        band_cols = [c for c in df.columns
                     if c.startswith(f"{sig}_bp_") and not c.endswith("_rel")]
        if not band_cols:
            continue
        stats_rows = []
        for c in CLASSES:
            sub = train[train["class"] == c]
            for bc in band_cols:
                stats_rows.append({
                    "class": c,
                    "band": bc.replace(f"{sig}_bp_", ""),
                    "mean": sub[bc].mean(),
                })
        sdf = pd.DataFrame(stats_rows)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bands = sdf["band"].unique().tolist()
        x = np.arange(len(bands))
        w = 0.27
        for i, c in enumerate(CLASSES):
            vals = [sdf.loc[(sdf["class"] == c) & (sdf["band"] == b), "mean"].values
                    for b in bands]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (i - 1) * w, vals, w, color=CLASS_PALETTE[c], label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.set_ylabel("Mean band power")
        ax.set_yscale("log")
        ax.set_title(f"Band powers per class — {sig}")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"band_power_bar_{sig}.png", dpi=130)
        plt.close(fig)


def plot_spectral_entropy_violin(df: pd.DataFrame) -> None:
    train = df[df["split"] == "train"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=False)
    for ax, sig in zip(axes, SIGNALS):
        col = f"{sig}_spec_entropy"
        if col not in train.columns:
            continue
        d = train[["class", col]].dropna().rename(columns={col: "val"})
        sns.violinplot(data=d, x="class", y="val", ax=ax,
                       palette=CLASS_PALETTE, order=list(CLASSES), cut=0, inner="quartile")
        ax.set_title(sig)
        ax.set_xlabel("")
        ax.set_ylabel("Spectral entropy" if sig == SIGNALS[0] else "")
    fig.suptitle("Spectral entropy per class")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "spectral_entropy_violin.png", dpi=130)
    plt.close(fig)


def plot_hjorth_scatter(df: pd.DataFrame) -> None:
    train = df[df["split"] == "train"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, sig in zip(axes.flat, SIGNALS):
        mob = f"{sig}_hjorth_mobility"
        cmp_ = f"{sig}_hjorth_complexity"
        if mob not in train.columns or cmp_ not in train.columns:
            continue
        for c in CLASSES:
            sub = train[train["class"] == c]
            ax.scatter(sub[mob], sub[cmp_], s=9, alpha=0.45,
                       color=CLASS_PALETTE[c], label=c, edgecolors="none")
        ax.set_xlabel("Hjorth mobility")
        ax.set_ylabel("Hjorth complexity")
        ax.set_title(sig)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Hjorth mobility vs complexity — train")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hjorth_scatter.png", dpi=130)
    plt.close(fig)


def plot_top16_boxes(df: pd.DataFrame, anova: pd.DataFrame) -> None:
    train = df[df["split"] == "train"]
    top = anova.dropna(subset=["F"]).head(16)["feature"].tolist()
    if not top:
        return
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for ax, feat in zip(axes.flat, top):
        d = train[["class", feat]].dropna().rename(columns={feat: "val"})
        sns.boxplot(data=d, x="class", y="val", ax=ax,
                    palette=CLASS_PALETTE, order=list(CLASSES), showfliers=False)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)
    # hide any unused axes
    for ax in axes.flat[len(top):]:
        ax.axis("off")
    fig.suptitle("Top 16 ANOVA-F features — train", y=1.0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top16_tf_features_box.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sns.set_theme(context="notebook", style="whitegrid")

    all_parts: list[pd.DataFrame] = []
    all_fails: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        df_split, fails = process_split(split)
        all_parts.append(df_split)
        all_fails[split] = fails

    df = pd.concat(all_parts, ignore_index=True, sort=False)
    meta_cols = ["split", "subject", "class", "segment_idx", "segment_id"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + sorted(feature_cols)]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Count NaNs per feature (failure proxy)
    n_total = len(df)
    nan_counts = df[feature_cols].isna().sum()
    total_missing = int(nan_counts.sum())
    fully_failed = int((nan_counts == n_total).sum())

    # Save parquet
    out_parquet = TABLES_DIR / "tf_features.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"[save] {out_parquet} rows={len(df)} feats={len(feature_cols)}")

    # Dictionary
    dict_df = build_dictionary(feature_cols)
    dict_df.to_csv(TABLES_DIR / "tf_features_dictionary.csv", index=False)
    print(f"[save] tf_features_dictionary.csv  ({len(dict_df)} rows)")

    # Class means (both splits)
    cm = class_means_table(df, feature_cols)
    cm.to_csv(TABLES_DIR / "tf_features_class_means.csv", index=False)
    print(f"[save] tf_features_class_means.csv")

    # ANOVA on subject-means (train)
    train_df = df[df["split"] == "train"].copy()
    anova = anova_table(train_df, feature_cols)
    anova.to_csv(TABLES_DIR / "tf_features_anova_train.csv", index=False)
    print(f"[save] tf_features_anova_train.csv")

    # Plots
    print("[plots] generating ...")
    tensor_train, meta_train = load_split("train")
    plot_psd_mean_per_class(tensor_train, meta_train, SFREQ)
    plot_band_power_bars(df)
    plot_spectral_entropy_violin(df)
    plot_hjorth_scatter(df)
    plot_top16_boxes(df, anova)

    # Sanity statistics
    def _mean_safe(col: str) -> float:
        return float(df[df["split"] == "train"][col].dropna().mean()) if col in df else np.nan

    sanity = {
        "BVP_peak_freq_mean_Hz": _mean_safe("Bvp_peak_freq"),
        "BVP_dom_freq_cardiac_Hz": _mean_safe("Bvp_dom_freq_cardiac"),
        "BVP_lf_hf_ratio_mean": _mean_safe("Bvp_lf_hf_ratio"),
        "RESP_dom_freq_breathing_Hz": _mean_safe("Resp_dom_freq_breathing"),
        "RESP_peak_freq_Hz": _mean_safe("Resp_peak_freq"),
        "EDA_spec_centroid_Hz": _mean_safe("Eda_spec_centroid"),
        "SpO2_total_variance": _mean_safe("SpO2_total_variance"),
    }

    # Report
    top15 = anova.dropna(subset=["F"]).head(15).copy()
    top15["p"] = top15["p"].map(lambda v: f"{v:.3g}")
    top15["F"] = top15["F"].map(lambda v: f"{v:.3f}")

    report_lines: list[str] = []
    report_lines.append("# Time + frequency-domain feature extraction — summary\n")
    report_lines.append(f"- Segments: train={int((df['split'] == 'train').sum())}, "
                        f"validation={int((df['split'] == 'validation').sum())}")
    report_lines.append(f"- Features per segment: {len(feature_cols)} "
                        f"(4 signals × ~{len(feature_cols) // 4})")
    report_lines.append(f"- Missing values: {total_missing} NaNs across table "
                        f"({fully_failed} features never extracted)")
    report_lines.append("")
    report_lines.append("## Top 15 features by ANOVA F (subject-mean, 3 classes, train)\n")
    report_lines.append(top15[["feature", "F", "p", "n"]].to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## Sanity checks (train means)\n")
    for k, v in sanity.items():
        report_lines.append(f"- {k}: {v:.4f}")
    report_lines.append("")
    # Interpretations
    def _cls_mean(feat: str, cls: str) -> float:
        sub = df[(df["split"] == "train") & (df["class"] == cls)]
        if feat not in sub.columns:
            return np.nan
        return float(sub[feat].dropna().mean())

    bvp_lfhf = {c: _cls_mean("Bvp_lf_hf_ratio", c) for c in CLASSES}
    resp_dom = {c: _cls_mean("Resp_dom_freq_breathing", c) for c in CLASSES}
    eda_total = {c: _cls_mean("Eda_total_power", c) for c in CLASSES}

    report_lines.append("## Interpretation\n")
    if all(np.isfinite(list(bvp_lfhf.values()))):
        direction = "increase" if (bvp_lfhf["PainArm"] + bvp_lfhf["PainHand"]) / 2 > bvp_lfhf["NoPain"] else "decrease"
        report_lines.append(
            f"- BVP LF/HF ratio: NoPain={bvp_lfhf['NoPain']:.2f}, "
            f"PainArm={bvp_lfhf['PainArm']:.2f}, PainHand={bvp_lfhf['PainHand']:.2f}. "
            f"Mean {direction} during pain suggests sympathetic shift."
        )
    if all(np.isfinite(list(resp_dom.values()))):
        report_lines.append(
            f"- RESP dominant breathing frequency: NoPain={resp_dom['NoPain']:.3f} Hz, "
            f"PainArm={resp_dom['PainArm']:.3f} Hz, PainHand={resp_dom['PainHand']:.3f} Hz "
            f"(~{60 * resp_dom['NoPain']:.1f}, {60 * resp_dom['PainArm']:.1f}, "
            f"{60 * resp_dom['PainHand']:.1f} breaths/min)."
        )
    if all(np.isfinite(list(eda_total.values()))):
        direction = "higher" if (eda_total["PainArm"] + eda_total["PainHand"]) / 2 > eda_total["NoPain"] else "lower"
        report_lines.append(
            f"- EDA total spectral power: NoPain={eda_total['NoPain']:.2g}, "
            f"PainArm={eda_total['PainArm']:.2g}, PainHand={eda_total['PainHand']:.2g} "
            f"— pain trials {direction}, consistent with increased sweat-gland activity."
        )
    report_lines.append("")
    report_lines.append("## Failed extractions per split\n")
    for split, fails in all_fails.items():
        if fails:
            report_lines.append(f"- {split}: {fails}")
        else:
            report_lines.append(f"- {split}: 0 segment-level failures")
    report_lines.append("")
    report_lines.append("## Outputs\n")
    report_lines.append(f"- `{out_parquet}`")
    report_lines.append("- `results/tables/tf_features_dictionary.csv`")
    report_lines.append("- `results/tables/tf_features_class_means.csv`")
    report_lines.append("- `results/tables/tf_features_anova_train.csv`")
    report_lines.append("- `plots/tfdomain/*.png`")

    rep_fp = REPORTS_DIR / "04_tfdomain_features_summary.md"
    rep_fp.write_text("\n".join(report_lines))
    print(f"[save] {rep_fp}")
    print("Done.")


if __name__ == "__main__":
    main()
