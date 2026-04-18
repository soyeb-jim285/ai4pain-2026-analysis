"""Physiological-feature extraction for the AI4Pain 2026 dataset.

Per-segment (10 s @ 100 Hz) features across BVP, EDA, RESP, SpO2 plus
one cross-signal feature (BVP-RESP coupling proxy).

Outputs tables, plots, and a summary report under results/ and plots/physio/.
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as spsig
from scipy import stats as spstats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import CLASSES, SFREQ, SIGNALS, load_split  # noqa: E402

warnings.filterwarnings("ignore")

try:
    import neurokit2 as nk  # noqa: F401
    HAVE_NK = True
except Exception:  # pragma: no cover
    HAVE_NK = False

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_DIR / "results" / "tables"
REPORTS_DIR = ANALYSIS_DIR / "results" / "reports"
PLOTS_DIR = ANALYSIS_DIR / "plots" / "physio"
for d in (TABLES_DIR, REPORTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Extraction-status tracking
# ---------------------------------------------------------------------------
STATUS: dict[str, Counter] = {}


def _mark(feature: str, status: str) -> None:
    c = STATUS.setdefault(feature, Counter())
    c[status] += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_nan(x: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(x)
    if not mask.any():
        return np.array([], dtype=np.float32)
    # trailing NaNs are padding; keep through last non-NaN
    last = int(np.flatnonzero(mask)[-1]) + 1
    x = x[:last]
    # if there are still NaNs inside, linear-interpolate
    if np.isnan(x).any():
        idx = np.arange(len(x))
        good = ~np.isnan(x)
        if good.sum() < 2:
            return np.array([], dtype=np.float32)
        x = np.interp(idx, idx[good], x[good]).astype(np.float32)
    return x


def _butter_filtfilt(
    x: np.ndarray,
    low: float | None,
    high: float | None,
    sfreq: float,
    order: int = 3,
) -> np.ndarray:
    nyq = 0.5 * sfreq
    if low is not None and high is not None:
        b, a = spsig.butter(order, [low / nyq, high / nyq], btype="band")
    elif low is not None:
        b, a = spsig.butter(order, low / nyq, btype="high")
    elif high is not None:
        b, a = spsig.butter(order, high / nyq, btype="low")
    else:
        return x
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    if padlen < 1:
        return x
    return spsig.filtfilt(b, a, x, padlen=padlen)


def _safe_stat(fn, *args, **kwargs) -> float:
    try:
        val = fn(*args, **kwargs)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        return float(val)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# BVP (PPG)
# ---------------------------------------------------------------------------
def extract_bvp_features(x: np.ndarray, sfreq: float = SFREQ) -> dict:
    out = {
        "BVP_hr_mean": np.nan,
        "BVP_peak_count": np.nan,
        "BVP_rr_mean": np.nan,
        "BVP_sdnn": np.nan,
        "BVP_rmssd": np.nan,
        "BVP_pnn20": np.nan,
        "BVP_peak_amp_mean": np.nan,
        "BVP_peak_amp_std": np.nan,
        "BVP_risetime_mean": np.nan,
        "BVP_env_std": np.nan,
    }
    x = _strip_nan(x)
    if len(x) < int(sfreq * 2):
        _mark("BVP_hr_mean", "nan")
        return out
    try:
        xf = _butter_filtfilt(x, 0.5, 8.0, sfreq, order=3)
    except Exception:
        xf = x - np.mean(x)

    peaks_idx = None
    # neurokit2 peaks first
    if HAVE_NK:
        try:
            _, info = nk.ppg_peaks(xf, sampling_rate=int(sfreq), show=False)
            peaks_idx = np.asarray(info.get("PPG_Peaks", []), dtype=int)
            if peaks_idx.size < 2:
                peaks_idx = None
            else:
                _mark("BVP_hr_mean", "ok")
        except Exception:
            peaks_idx = None
    if peaks_idx is None:
        try:
            min_dist = int(sfreq * 0.35)  # ~170 bpm max
            prom = np.nanstd(xf) * 0.3
            peaks_idx, _ = spsig.find_peaks(xf, distance=min_dist, prominence=prom)
            if peaks_idx.size >= 2:
                _mark("BVP_hr_mean", "fallback")
            else:
                _mark("BVP_hr_mean", "nan")
        except Exception:
            _mark("BVP_hr_mean", "nan")
            return out

    if peaks_idx is None or peaks_idx.size < 2:
        return out

    out["BVP_peak_count"] = float(peaks_idx.size)
    rr = np.diff(peaks_idx) / sfreq  # seconds
    if rr.size == 0:
        return out
    rr_ms = rr * 1000.0

    out["BVP_rr_mean"] = float(np.mean(rr))
    # HR bpm
    out["BVP_hr_mean"] = float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else np.nan
    out["BVP_sdnn"] = float(np.std(rr_ms, ddof=1)) if rr.size > 1 else np.nan
    if rr.size > 1:
        diff = np.diff(rr_ms)
        out["BVP_rmssd"] = float(np.sqrt(np.mean(diff**2)))
        out["BVP_pnn20"] = float(np.mean(np.abs(diff) > 20.0))
    else:
        out["BVP_rmssd"] = np.nan
        out["BVP_pnn20"] = np.nan

    # peak amplitudes
    amps = xf[peaks_idx]
    out["BVP_peak_amp_mean"] = float(np.mean(amps))
    out["BVP_peak_amp_std"] = float(np.std(amps)) if amps.size > 1 else 0.0

    # rise time: from preceding trough to peak
    try:
        troughs_idx, _ = spsig.find_peaks(-xf, distance=int(sfreq * 0.3))
        rise_times = []
        for pk in peaks_idx:
            prev_tr = troughs_idx[troughs_idx < pk]
            if prev_tr.size:
                rise_times.append((pk - prev_tr[-1]) / sfreq)
        out["BVP_risetime_mean"] = (
            float(np.mean(rise_times)) if rise_times else np.nan
        )
    except Exception:
        out["BVP_risetime_mean"] = np.nan

    # envelope std
    try:
        analytic = spsig.hilbert(xf)
        out["BVP_env_std"] = float(np.std(np.abs(analytic)))
    except Exception:
        out["BVP_env_std"] = np.nan
    return out


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def extract_eda_features(x: np.ndarray, sfreq: float = SFREQ) -> dict:
    out = {
        "EDA_tonic_mean": np.nan,
        "EDA_tonic_slope": np.nan,
        "EDA_scr_count": np.nan,
        "EDA_scr_amp_mean": np.nan,
        "EDA_scr_rate": np.nan,
        "EDA_phasic_auc": np.nan,
        "EDA_range": np.nan,
    }
    x = _strip_nan(x)
    if len(x) < int(sfreq * 2):
        _mark("EDA_tonic_mean", "nan")
        return out

    duration_s = len(x) / sfreq
    out["EDA_range"] = float(np.ptp(x))

    # Tonic via lowpass
    try:
        tonic = _butter_filtfilt(x, None, 0.05, sfreq, order=2)
        out["EDA_tonic_mean"] = float(np.mean(tonic))
        t = np.arange(len(tonic)) / sfreq
        slope, _ = np.polyfit(t, tonic, 1)
        out["EDA_tonic_slope"] = float(slope)
        _mark("EDA_tonic_mean", "ok")
    except Exception:
        tonic = np.full_like(x, np.mean(x))
        out["EDA_tonic_mean"] = float(np.mean(x))
        out["EDA_tonic_slope"] = np.nan
        _mark("EDA_tonic_mean", "fallback")

    # Phasic: try neurokit2, fall back to x - tonic
    phasic = None
    if HAVE_NK:
        try:
            df = nk.eda_phasic(
                nk.signal_sanitize(x), sampling_rate=int(sfreq), method="highpass"
            )
            if "EDA_Phasic" in df.columns:
                phasic = df["EDA_Phasic"].to_numpy()
                _mark("EDA_scr_count", "ok")
        except Exception:
            phasic = None
    if phasic is None:
        try:
            phasic = _butter_filtfilt(x, 0.05, None, sfreq, order=2)
            _mark("EDA_scr_count", "fallback")
        except Exception:
            phasic = x - tonic
            _mark("EDA_scr_count", "fallback")

    # SCR peak detection
    try:
        prom = max(0.01, 0.1 * np.nanstd(phasic))
        peaks_idx, props = spsig.find_peaks(
            phasic, prominence=prom, distance=int(sfreq * 1.0)
        )
        out["EDA_scr_count"] = float(peaks_idx.size)
        out["EDA_scr_rate"] = float(peaks_idx.size / (duration_s / 10.0))
        if peaks_idx.size:
            out["EDA_scr_amp_mean"] = float(np.mean(props["prominences"]))
        else:
            out["EDA_scr_amp_mean"] = 0.0
    except Exception:
        pass

    try:
        out["EDA_phasic_auc"] = float(np.trapz(np.abs(phasic)) / sfreq)
    except Exception:
        out["EDA_phasic_auc"] = np.nan
    return out


# ---------------------------------------------------------------------------
# RESP
# ---------------------------------------------------------------------------
def extract_resp_features(x: np.ndarray, sfreq: float = SFREQ) -> dict:
    out = {
        "RESP_rate": np.nan,
        "RESP_interval_std": np.nan,
        "RESP_amp_mean": np.nan,
        "RESP_amp_std": np.nan,
        "RESP_ie_ratio": np.nan,
    }
    x = _strip_nan(x)
    if len(x) < int(sfreq * 2):
        _mark("RESP_rate", "nan")
        return out
    try:
        xf = _butter_filtfilt(x, 0.1, 0.5, sfreq, order=2)
        _mark("RESP_rate", "ok")
    except Exception:
        xf = x - np.mean(x)
        _mark("RESP_rate", "fallback")

    duration_s = len(x) / sfreq
    try:
        min_dist = int(sfreq * 1.0)  # ~60/min max
        prom = max(1e-6, 0.1 * np.nanstd(xf))
        peaks, props = spsig.find_peaks(xf, distance=min_dist, prominence=prom)
        troughs, _ = spsig.find_peaks(-xf, distance=min_dist, prominence=prom)
        if peaks.size >= 2:
            intervals = np.diff(peaks) / sfreq
            out["RESP_rate"] = float(60.0 / np.mean(intervals))
            out["RESP_interval_std"] = (
                float(np.std(intervals)) if intervals.size > 1 else 0.0
            )
        elif peaks.size == 1:
            out["RESP_rate"] = float(60.0 / duration_s)
            out["RESP_interval_std"] = np.nan
        else:
            out["RESP_rate"] = 0.0
            out["RESP_interval_std"] = np.nan

        if peaks.size:
            amps = xf[peaks] - (xf[troughs].mean() if troughs.size else 0.0)
            out["RESP_amp_mean"] = float(np.mean(amps))
            out["RESP_amp_std"] = float(np.std(amps)) if amps.size > 1 else 0.0

        # Inhale / exhale ratio proxy:
        # For each peak, find the preceding trough (inhale = trough->peak)
        # and following trough (exhale = peak->trough). Median ratio.
        ratios = []
        for pk in peaks:
            prev_tr = troughs[troughs < pk]
            next_tr = troughs[troughs > pk]
            if prev_tr.size and next_tr.size:
                inh = (pk - prev_tr[-1]) / sfreq
                exh = (next_tr[0] - pk) / sfreq
                if exh > 0:
                    ratios.append(inh / exh)
        if ratios:
            out["RESP_ie_ratio"] = float(np.median(ratios))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# SpO2
# ---------------------------------------------------------------------------
def extract_spo2_features(x: np.ndarray, sfreq: float = SFREQ) -> dict:
    out = {
        "SPO2_mean": np.nan,
        "SPO2_std": np.nan,
        "SPO2_min": np.nan,
        "SPO2_dip_magnitude": np.nan,
        "SPO2_dip_count": np.nan,
        "SPO2_slope": np.nan,
    }
    x = _strip_nan(x)
    if len(x) < 10:
        _mark("SPO2_mean", "nan")
        return out
    try:
        mean_v = float(np.mean(x))
        out["SPO2_mean"] = mean_v
        out["SPO2_std"] = float(np.std(x))
        out["SPO2_min"] = float(np.min(x))
        out["SPO2_dip_magnitude"] = mean_v - float(np.min(x))
        # dips > 2% below mean via trough detection
        troughs, _ = spsig.find_peaks(-x, distance=int(sfreq * 0.5))
        out["SPO2_dip_count"] = float(
            int(np.sum((mean_v - x[troughs]) > 2.0)) if troughs.size else 0
        )
        t = np.arange(len(x)) / sfreq
        slope, _ = np.polyfit(t, x, 1)
        out["SPO2_slope"] = float(slope)
        _mark("SPO2_mean", "ok")
    except Exception:
        _mark("SPO2_mean", "fallback")
    return out


# ---------------------------------------------------------------------------
# Cross-signal: BVP envelope x RESP
# ---------------------------------------------------------------------------
def extract_cross_features(
    bvp: np.ndarray, resp: np.ndarray, sfreq: float = SFREQ
) -> dict:
    out = {"BVP_RESP_envcorr": np.nan}
    bvp = _strip_nan(bvp)
    resp = _strip_nan(resp)
    n = min(len(bvp), len(resp))
    if n < int(sfreq * 3):
        _mark("BVP_RESP_envcorr", "nan")
        return out
    try:
        bvp_f = _butter_filtfilt(bvp[:n], 0.5, 8.0, sfreq, order=3)
        resp_f = _butter_filtfilt(resp[:n], 0.1, 0.5, sfreq, order=2)
        env = np.abs(spsig.hilbert(bvp_f))
        # match lengths, then remove mean
        env = env - np.mean(env)
        rf = resp_f - np.mean(resp_f)
        if np.std(env) < 1e-12 or np.std(rf) < 1e-12:
            _mark("BVP_RESP_envcorr", "nan")
            return out
        r, _ = spstats.pearsonr(env, rf)
        out["BVP_RESP_envcorr"] = float(r)
        _mark("BVP_RESP_envcorr", "ok")
    except Exception:
        _mark("BVP_RESP_envcorr", "fallback")
    return out


# ---------------------------------------------------------------------------
# Feature dictionary
# ---------------------------------------------------------------------------
FEATURE_DICT = [
    ("BVP_hr_mean", "BVP", "Mean heart rate (bpm) from peak intervals", "40-180 bpm", "peak detection (nk.ppg_peaks or scipy)"),
    ("BVP_peak_count", "BVP", "Number of PPG peaks in the segment", "5-25", "peak detection"),
    ("BVP_rr_mean", "BVP", "Mean peak-to-peak interval (s)", "0.33-1.5", "diff of peaks / sfreq"),
    ("BVP_sdnn", "BVP", "Std of RR intervals (ms)", "5-200", "std(RR_ms)"),
    ("BVP_rmssd", "BVP", "Root mean square of successive RR differences (ms)", "5-200", "sqrt(mean(diff(RR_ms)^2))"),
    ("BVP_pnn20", "BVP", "Proxy pNN20: fraction of |diff(RR)|>20ms (proxy for short windows)", "0-1", "mean(|diff(RR_ms)|>20)"),
    ("BVP_peak_amp_mean", "BVP", "Mean peak amplitude after bandpass filter", "unitless", "mean(x_bp[peaks])"),
    ("BVP_peak_amp_std", "BVP", "Std of peak amplitudes", ">=0", "std(x_bp[peaks])"),
    ("BVP_risetime_mean", "BVP", "Mean time from trough to following peak (s)", "0.05-0.5", "peak idx - prev trough idx"),
    ("BVP_env_std", "BVP", "Std of Hilbert envelope of bandpassed BVP", ">=0", "std(|hilbert(x_bp)|)"),
    ("EDA_tonic_mean", "EDA", "Mean tonic level (µS) via <0.05 Hz lowpass", "0.1-30", "mean(lowpass(x,0.05))"),
    ("EDA_tonic_slope", "EDA", "Linear slope of tonic level over segment (µS/s)", "-2..2", "polyfit tonic vs t"),
    ("EDA_scr_count", "EDA", "Number of SCR peaks on phasic signal", "0-10", "scipy find_peaks on phasic"),
    ("EDA_scr_amp_mean", "EDA", "Mean prominence of SCR peaks", ">=0", "mean(peak prominences)"),
    ("EDA_scr_rate", "EDA", "SCR peaks per 10 s", "0-10", "peaks / (duration/10)"),
    ("EDA_phasic_auc", "EDA", "AUC of |phasic| over time (µS*s)", ">=0", "trapz(|phasic|)/sfreq"),
    ("EDA_range", "EDA", "Peak-to-peak range of EDA (µS)", ">=0", "ptp(x)"),
    ("RESP_rate", "RESP", "Respiratory rate (breaths/min)", "6-40", "60 / mean peak interval after 0.1-0.5 Hz bandpass"),
    ("RESP_interval_std", "RESP", "Std of peak-to-peak intervals (s)", ">=0", "std(diff(peaks)/sfreq)"),
    ("RESP_amp_mean", "RESP", "Mean peak amplitude (peak - mean trough)", "unitless", "mean(xf[peaks]-mean(xf[troughs]))"),
    ("RESP_amp_std", "RESP", "Std of peak amplitudes", ">=0", "std(xf[peaks])"),
    ("RESP_ie_ratio", "RESP", "Inhale:exhale ratio proxy (median)", "0.3-2", "median(inh/exh per cycle)"),
    ("SPO2_mean", "SpO2", "Mean SpO2 (%)", "90-100", "mean(x)"),
    ("SPO2_std", "SpO2", "Std of SpO2 (%)", ">=0", "std(x)"),
    ("SPO2_min", "SpO2", "Minimum SpO2 in segment (%)", "80-100", "min(x)"),
    ("SPO2_dip_magnitude", "SpO2", "Mean - min SpO2 (%)", ">=0", "mean(x) - min(x)"),
    ("SPO2_dip_count", "SpO2", "Number of troughs >2% below mean", "0-5", "scipy find_peaks on -x"),
    ("SPO2_slope", "SpO2", "Linear slope of SpO2 over segment (%/s)", "-1..1", "polyfit vs t"),
    ("BVP_RESP_envcorr", "Cross", "Pearson correlation between |hilbert(BVP_bp)| and RESP_bp", "-1..1", "pearsonr(env, resp)"),
]
FEATURE_ORDER = [row[0] for row in FEATURE_DICT]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def extract_all(tensor: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sig_idx = {s: i for i, s in enumerate(SIGNALS)}
    for i in tqdm(range(tensor.shape[0]), desc="segments"):
        seg = tensor[i]
        feats: dict[str, float] = {}
        try:
            feats.update(extract_bvp_features(seg[sig_idx["Bvp"]], SFREQ))
        except Exception:
            for k in FEATURE_ORDER:
                if k.startswith("BVP_") and not k.startswith("BVP_RESP"):
                    feats.setdefault(k, np.nan)
        try:
            feats.update(extract_eda_features(seg[sig_idx["Eda"]], SFREQ))
        except Exception:
            for k in FEATURE_ORDER:
                if k.startswith("EDA_"):
                    feats.setdefault(k, np.nan)
        try:
            feats.update(extract_resp_features(seg[sig_idx["Resp"]], SFREQ))
        except Exception:
            for k in FEATURE_ORDER:
                if k.startswith("RESP_"):
                    feats.setdefault(k, np.nan)
        try:
            feats.update(extract_spo2_features(seg[sig_idx["SpO2"]], SFREQ))
        except Exception:
            for k in FEATURE_ORDER:
                if k.startswith("SPO2_"):
                    feats.setdefault(k, np.nan)
        try:
            feats.update(
                extract_cross_features(
                    seg[sig_idx["Bvp"]], seg[sig_idx["Resp"]], SFREQ
                )
            )
        except Exception:
            feats["BVP_RESP_envcorr"] = np.nan

        for k in FEATURE_ORDER:
            feats.setdefault(k, np.nan)
        rows.append(feats)

    feat_df = pd.DataFrame(rows, columns=FEATURE_ORDER)
    meta_cols = ["split", "subject", "class", "segment_idx", "segment_id"]
    out = pd.concat(
        [meta[meta_cols].reset_index(drop=True), feat_df.reset_index(drop=True)],
        axis=1,
    )
    return out


def status_frame() -> pd.DataFrame:
    rows = []
    for feat in FEATURE_ORDER:
        c = STATUS.get(feat, Counter())
        rows.append(
            {
                "feature": feat,
                "count_ok": int(c.get("ok", 0)),
                "count_fallback": int(c.get("fallback", 0)),
                "count_nan": int(c.get("nan", 0)),
            }
        )
    return pd.DataFrame(rows)


def class_means(df: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in df.columns if c in FEATURE_ORDER]
    agg = df.groupby(["split", "class"])[features].agg(["mean", "std"])
    agg.columns = [f"{f}_{stat}" for f, stat in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_outputs(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    train = df[df["split"] == "train"].copy()

    # Feature boxplots: 16 most-varying (by class-level std of class means)
    feats = [c for c in df.columns if c in FEATURE_ORDER]
    class_means_train = train.groupby("class")[feats].mean()
    # variability across classes: std of per-class means, then normalised by feature std
    feature_std = train[feats].std(ddof=0).replace(0, np.nan)
    variability = (
        class_means_train.std(axis=0, ddof=0) / feature_std
    ).fillna(0).sort_values(ascending=False)
    top16 = variability.head(16).index.tolist()

    n = len(top16)
    nrow = 4
    ncol = 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = axes.flatten()
    for ax, feat in zip(axes, top16):
        data = [train.loc[train["class"] == c, feat].dropna() for c in CLASSES]
        ax.boxplot(data, labels=CLASSES, showfliers=False)
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Top-16 class-variable features (train)", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_boxplots_by_class.png", dpi=130)
    plt.close(fig)

    # HR histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in CLASSES:
        vals = train.loc[train["class"] == c, "BVP_hr_mean"].dropna()
        ax.hist(vals, bins=40, alpha=0.55, label=c, density=True)
    ax.set_xlabel("Heart rate (bpm)")
    ax.set_ylabel("Density")
    ax.set_title("Estimated HR per class (train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bvp_hr_per_class.png", dpi=130)
    plt.close(fig)

    # SCR count violin
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=train, x="class", y="EDA_scr_count", order=list(CLASSES),
        ax=ax, cut=0, inner="quartile",
    )
    ax.set_title("EDA SCR count per class (train)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_scr_count_per_class.png", dpi=130)
    plt.close(fig)

    # Respiration rate violin
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=train, x="class", y="RESP_rate", order=list(CLASSES),
        ax=ax, cut=0, inner="quartile",
    )
    ax.set_title("Respiration rate per class (train)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "resp_rate_per_class.png", dpi=130)
    plt.close(fig)

    # SpO2 mean boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [train.loc[train["class"] == c, "SPO2_mean"].dropna() for c in CLASSES]
    ax.boxplot(data, labels=CLASSES, showfliers=True)
    ax.set_title("Mean SpO2 per class (train)")
    ax.set_ylabel("SpO2 (%)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "spo2_mean_per_class.png", dpi=130)
    plt.close(fig)

    # Example subjects HR over segments
    subjects = sorted(train["subject"].unique())[:6]
    for sid in subjects:
        sub = train[train["subject"] == sid].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(by=["class", "segment_idx"]).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = {"NoPain": "#4c72b0", "PainArm": "#dd8452", "PainHand": "#c44e52"}
        bar_colors = [colors[c] for c in sub["class"]]
        xs = np.arange(len(sub))
        ax.bar(xs, sub["BVP_hr_mean"].values, color=bar_colors)
        labels = [f"{c[:4]}-{i}" for c, i in zip(sub["class"], sub["segment_idx"])]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_ylabel("HR (bpm)")
        ax.set_title(f"Subject {sid} estimated HR per segment (ordered by class)")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in CLASSES
        ]
        ax.legend(handles, CLASSES, loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"example_subject_{sid}_hr_over_segments.png", dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def anova_top_features(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """ANOVA F on subject-level means (to reduce pseudoreplication)."""
    train = df[df["split"] == "train"]
    feats = [c for c in df.columns if c in FEATURE_ORDER]
    subj_means = train.groupby(["subject", "class"])[feats].mean().reset_index()
    rows = []
    for feat in feats:
        groups = [
            subj_means.loc[subj_means["class"] == c, feat].dropna().values
            for c in CLASSES
        ]
        if any(len(g) < 2 for g in groups):
            rows.append({"feature": feat, "F": np.nan, "p": np.nan,
                         **{f"mean_{c}": np.nan for c in CLASSES}})
            continue
        try:
            F, p = spstats.f_oneway(*groups)
        except Exception:
            F, p = np.nan, np.nan
        means = {
            f"mean_{c}": float(np.nanmean(groups[i])) for i, c in enumerate(CLASSES)
        }
        rows.append({"feature": feat, "F": float(F) if np.isfinite(F) else np.nan,
                     "p": float(p) if np.isfinite(p) else np.nan, **means})
    res = pd.DataFrame(rows)
    res = res.sort_values("F", ascending=False, na_position="last")
    return res.head(k).reset_index(drop=True)


def write_summary(
    df: pd.DataFrame, status_df: pd.DataFrame, top_anova: pd.DataFrame
) -> Path:
    train = df[df["split"] == "train"]
    n_total = len(df)
    n_train = len(train)

    reliability_lines = []
    for _, r in status_df.iterrows():
        tot = r["count_ok"] + r["count_fallback"] + r["count_nan"]
        if tot == 0:
            pct_ok = np.nan
        else:
            pct_ok = 100.0 * (r["count_ok"] + r["count_fallback"]) / tot
        reliability_lines.append(
            f"| {r['feature']} | {r['count_ok']} | {r['count_fallback']} | "
            f"{r['count_nan']} | {pct_ok:.1f}% |"
        )

    # Sanity checks (train)
    hr = train["BVP_hr_mean"].dropna()
    rr = train["RESP_rate"].dropna()
    spo2 = train["SPO2_mean"].dropna()
    sanity = []
    sanity.append(
        f"- HR median={hr.median():.1f} bpm (P5={hr.quantile(0.05):.1f}, "
        f"P95={hr.quantile(0.95):.1f}). "
        + ("OK" if 50 <= hr.median() <= 110 else "OUT OF EXPECTED 60-100 bpm RANGE")
    )
    sanity.append(
        f"- RESP median={rr.median():.1f} /min (P5={rr.quantile(0.05):.1f}, "
        f"P95={rr.quantile(0.95):.1f}). "
        + ("OK" if 6 <= rr.median() <= 30 else "OUT OF EXPECTED 10-20 /min RANGE")
    )
    sanity.append(
        f"- SpO2 median={spo2.median():.1f}% (P5={spo2.quantile(0.05):.1f}, "
        f"P95={spo2.quantile(0.95):.1f}). "
        + ("OK" if 90 <= spo2.median() <= 100 else "OUT OF EXPECTED 95-100% RANGE")
    )

    # Physio interpretation for top features
    interp = []
    for _, r in top_anova.iterrows():
        means = {c: r[f"mean_{c}"] for c in CLASSES}
        direction = []
        pain_mean = 0.5 * (means["PainArm"] + means["PainHand"])
        sign = "higher" if pain_mean > means["NoPain"] else "lower"
        feat = r["feature"]
        hint = ""
        if feat.startswith("EDA_scr"):
            hint = " (SCR activity reflects sympathetic arousal)"
        elif feat.startswith("BVP_hr") or feat == "BVP_rr_mean":
            hint = " (HR rise / RR shortening is classic for pain)"
        elif feat.startswith("BVP_rmssd") or feat.startswith("BVP_sdnn") or feat == "BVP_pnn20":
            hint = " (parasympathetic HRV typically drops with pain)"
        elif feat.startswith("RESP"):
            hint = " (breathing rate and variability shift with pain/stress)"
        elif feat.startswith("SPO2"):
            hint = " (peripheral vasoconstriction under pain can shift SpO2)"
        elif feat.startswith("EDA"):
            hint = " (EDA rises with sympathetic activation)"
        interp.append(
            f"- **{feat}** (F={r['F']:.2f}, p={r['p']:.1e}): Pain {sign} than NoPain"
            f" (NoPain={means['NoPain']:.3f}, PainArm={means['PainArm']:.3f}, "
            f"PainHand={means['PainHand']:.3f}).{hint}"
        )

    md = []
    md.append("# 03 Physiological Features Summary\n")
    md.append(f"Segments processed: {n_total} ({n_train} train + "
              f"{n_total - n_train} validation). Sampling rate: {SFREQ} Hz.\n")
    md.append("## Extraction reliability\n")
    md.append("Status is tracked per representative feature (other features in "
              "the same signal share the same extraction path and inherit the "
              "outcome for their segment). `ok` = main method, `fallback` = "
              "manual backup, `nan` = skipped (segment too short or error).\n")
    md.append("| feature | ok | fallback | nan | % success |")
    md.append("|---|---:|---:|---:|---:|")
    md.extend(reliability_lines)
    md.append("")
    md.append("## Top 10 class-discriminative features (train, subject-averaged ANOVA)\n")
    md.append("Computed on subject-level means (one row per (subject, class)) "
              "to reduce pseudoreplication.\n")
    md.append("| rank | feature | F | p | NoPain | PainArm | PainHand |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for i, r in top_anova.iterrows():
        md.append(
            f"| {i+1} | {r['feature']} | {r['F']:.3f} | {r['p']:.2e} | "
            f"{r['mean_NoPain']:.3f} | {r['mean_PainArm']:.3f} | "
            f"{r['mean_PainHand']:.3f} |"
        )
    md.append("")
    md.append("## Physiological interpretation\n")
    md.extend(interp)
    md.append("")
    md.append("## Sanity checks\n")
    md.extend(sanity)
    md.append("")
    md.append("## Outputs\n")
    md.append("- `results/tables/physio_features.parquet`")
    md.append("- `results/tables/physio_features_dictionary.csv`")
    md.append("- `results/tables/physio_features_class_means.csv`")
    md.append("- `results/tables/physio_features_extraction_status.csv`")
    md.append("- `plots/physio/feature_boxplots_by_class.png`")
    md.append("- `plots/physio/bvp_hr_per_class.png`")
    md.append("- `plots/physio/eda_scr_count_per_class.png`")
    md.append("- `plots/physio/resp_rate_per_class.png`")
    md.append("- `plots/physio/spo2_mean_per_class.png`")
    md.append("- `plots/physio/example_subject_{sid}_hr_over_segments.png` (x6)")
    fp = REPORTS_DIR / "03_physio_features_summary.md"
    fp.write_text("\n".join(md))
    return fp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    frames = []
    for split in ("train", "validation"):
        print(f"[load] {split}")
        tensor, meta = load_split(split)
        print(f"  tensor {tensor.shape}  meta {len(meta)}")
        df = extract_all(tensor, meta)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    # Write feature table
    out_parquet = TABLES_DIR / "physio_features.parquet"
    full.to_parquet(out_parquet, index=False)
    print(f"[write] {out_parquet}  rows={len(full)}")

    # Dictionary
    dict_df = pd.DataFrame(
        FEATURE_DICT,
        columns=["feature_name", "signal", "description", "expected_range",
                 "extraction_method"],
    )
    dict_fp = TABLES_DIR / "physio_features_dictionary.csv"
    dict_df.to_csv(dict_fp, index=False)
    print(f"[write] {dict_fp}")

    # Class means
    cm = class_means(full)
    cm_fp = TABLES_DIR / "physio_features_class_means.csv"
    cm.to_csv(cm_fp, index=False)
    print(f"[write] {cm_fp}")

    # Extraction status
    st = status_frame()
    st_fp = TABLES_DIR / "physio_features_extraction_status.csv"
    st.to_csv(st_fp, index=False)
    print(f"[write] {st_fp}")

    # Plots
    print("[plot] generating figures")
    plot_outputs(full)

    # ANOVA + summary
    print("[summary] ranking features")
    top_anova = anova_top_features(full, k=10)
    sm_fp = write_summary(full, st, top_anova)
    print(f"[write] {sm_fp}")
    print("Done.")


if __name__ == "__main__":
    main()
