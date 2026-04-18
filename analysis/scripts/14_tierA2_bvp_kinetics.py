"""Tier-A2: BVP proximity kinetics for AI4Pain 2026 (PainArm vs PainHand).

Hypothesis
----------
BVP is recorded at the fingertip. TENS on the **hand** is electrically close
to the sensor (local cutaneous vasoconstriction); TENS on the **arm** is
proximal -> diffuse, delayed vasoconstriction. Earlier aggregate / morphology
/ reactivity passes (0/243 and 0/52 surviving FDR) did NOT include this
timing-kinetics angle. Decay rate of pulse amplitude and local
vasoconstriction markers should differ between arm and hand stimulation
sites if the proximity hypothesis holds.

This script
- builds 13 per-segment kinetics features (raw + Delta-from-baseline),
- runs paired Wilcoxon ARM vs HAND across 41 train subjects with BH-FDR,
- corroborates with a mixed-effects LMM,
- checks validation-split direction preservation,
- emits diagnostic plots and a markdown summary.

Run: uv run python scripts/14_tierA2_bvp_kinetics.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as spsig
from scipy import stats as spstats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import SFREQ, SIGNALS, load_split  # noqa: E402

warnings.filterwarnings("ignore")
np.random.seed(42)
RNG = np.random.default_rng(42)

try:
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAVE_LMM = True
except Exception:
    HAVE_LMM = False

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_DIR / "results" / "tables"
REPORTS_DIR = ANALYSIS_DIR / "results" / "reports"
PLOTS_DIR = ANALYSIS_DIR / "plots" / "tierA2_bvp_kinetics"
for d in (TABLES_DIR, REPORTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
BVP_IDX = SIGNALS.index("Bvp")
EDA_IDX = SIGNALS.index("Eda")
SPO2_IDX = SIGNALS.index("SpO2")

EPS = 1e-12

FEATURE_NAMES = [
    "bvp_amp_halfdecay_time_s",
    "bvp_amp_quartdecay_time_s",
    "bvp_onset_latency_vs_eda_s",
    "bvp_local_vasoconstriction_index",
    "bvp_beat_amp_trend_diff",
    "bvp_beat_amp_trend_ratio",
    "bvp_local_vasoconstriction_speed_slope",
    "bvp_local_vasoconstriction_speed_r2",
    "bvp_amplitude_jitter",
    "bvp_cv_interbeat",
    "bvp_recovery_halftime_s",
    "bvp_local_min_amp_s",
    "bvp_peak_envelope_slope",
    "bvp_peak_envelope_auc",
    "bvp_first_derivative_peak_time",
]

sns.set_context("talk", font_scale=0.7)
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------
# Helpers: signal conditioning
# ---------------------------------------------------------------------------
def _strip_nan(x: np.ndarray) -> np.ndarray:
    """Truncate trailing NaNs and linearly interpolate small interior gaps."""
    if x.size == 0:
        return x.astype(np.float32)
    mask = ~np.isnan(x)
    if not mask.any():
        return np.array([], dtype=np.float32)
    last = int(np.flatnonzero(mask)[-1]) + 1
    x = x[:last]
    if np.isnan(x).any():
        idx = np.arange(len(x))
        good = ~np.isnan(x)
        if good.sum() < 2:
            return np.array([], dtype=np.float32)
        x = np.interp(idx, idx[good], x[good])
    return x.astype(np.float32)


def _bandpass_bvp(x: np.ndarray, lo: float = 0.5, hi: float = 8.0,
                  fs: float = SFREQ, order: int = 3) -> np.ndarray:
    if len(x) < max(3 * order + 1, 15):
        return x.astype(np.float32)
    nyq = fs / 2.0
    try:
        b, a = spsig.butter(order, [lo / nyq, hi / nyq], btype="band")
        y = spsig.filtfilt(b, a, x, padlen=min(3 * max(len(a), len(b)), len(x) - 1))
    except Exception:
        return x.astype(np.float32)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extraction (per-segment)
# ---------------------------------------------------------------------------
def _windowed_max_amp(x: np.ndarray, fs: float, win_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_centres_s, max-abs amplitude per bin)."""
    win = max(1, int(round(win_s * fs)))
    n = len(x)
    n_bins = int(np.ceil(n / win))
    centres = np.zeros(n_bins, dtype=float)
    amps = np.zeros(n_bins, dtype=float)
    for k in range(n_bins):
        lo = k * win
        hi = min(n, lo + win)
        seg = x[lo:hi]
        if len(seg) == 0:
            amps[k] = np.nan
        else:
            amps[k] = float(np.max(np.abs(seg)))
        centres[k] = (lo + hi) / 2.0 / fs
    return centres, amps


def _decay_time(centres: np.ndarray, amps: np.ndarray, frac: float) -> float:
    """First time at which amps drops to <= frac * max(amps)."""
    if len(amps) == 0 or not np.any(np.isfinite(amps)):
        return np.nan
    mx = float(np.nanmax(amps))
    if mx <= 0 or not np.isfinite(mx):
        return np.nan
    target = frac * mx
    below = np.where(amps <= target)[0]
    if len(below) == 0:
        return np.nan
    return float(centres[below[0]])


def _per_beat_amplitudes(bvp_filt: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (peak_indices, beat_amplitudes peak-to-trough, peak_times_s)."""
    if len(bvp_filt) < int(1.5 * fs):
        return np.array([]), np.array([]), np.array([])
    min_dist = int(0.4 * fs)
    sd = float(np.std(bvp_filt))
    prom = 0.3 * sd if sd > 0 else None
    peaks, _ = spsig.find_peaks(bvp_filt, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return peaks, np.array([]), peaks / fs
    amps = np.full(len(peaks), np.nan, dtype=float)
    for i, pk in enumerate(peaks):
        # Trough searched in [prev_peak, pk] or [pk, next_peak]
        if i + 1 < len(peaks):
            seg = bvp_filt[pk:peaks[i + 1] + 1]
        else:
            seg = bvp_filt[pk:]
        if len(seg) < 2:
            continue
        tr_val = float(np.min(seg))
        amps[i] = float(bvp_filt[pk] - tr_val)
    return peaks, amps, peaks / fs


def _eda_first_slope_event_s(eda: np.ndarray, fs: float) -> float:
    """Time of first large EDA slope event (>=2 sigma above mean abs slope).

    Returns NaN if no event detected.
    """
    if len(eda) < int(2 * fs):
        return np.nan
    # Smooth EDA before differentiating
    try:
        eda_s = spsig.savgol_filter(eda, window_length=min(31, len(eda) // 2 * 2 + 1), polyorder=2)
    except Exception:
        eda_s = eda
    d = np.diff(eda_s) * fs
    abs_d = np.abs(d)
    mu = float(np.mean(abs_d))
    sd = float(np.std(abs_d))
    if sd <= 0:
        return np.nan
    thr = mu + 2.0 * sd
    above = np.where(abs_d > thr)[0]
    if len(above) == 0:
        return np.nan
    return float(above[0] / fs)


def _bvp_first_amp_drop_s(times: np.ndarray, amps: np.ndarray,
                          drop_frac: float = 0.15) -> float:
    """Time of first per-beat amplitude drop > drop_frac vs running max."""
    if len(amps) < 2:
        return np.nan
    finite = np.isfinite(amps)
    if finite.sum() < 2:
        return np.nan
    a = amps.copy()
    t = times.copy()
    a[~finite] = np.nan
    running_max = np.fmax.accumulate(np.nan_to_num(a, nan=-np.inf))
    running_max[running_max == -np.inf] = np.nan
    rel = (running_max - a) / (running_max + EPS)
    hits = np.where(rel > drop_frac)[0]
    if len(hits) == 0:
        return np.nan
    return float(t[hits[0]])


def _segment_window_amp(times: np.ndarray, amps: np.ndarray,
                        t_lo: float, t_hi: float) -> float:
    if len(amps) == 0:
        return np.nan
    mask = (times >= t_lo) & (times < t_hi) & np.isfinite(amps)
    if mask.sum() < 1:
        return np.nan
    return float(np.mean(amps[mask]))


def extract_kinetics(bvp_raw: np.ndarray, eda_raw: np.ndarray,
                     fs: float = SFREQ) -> dict[str, float]:
    """Compute the 13-feature BVP-kinetics dictionary for one segment."""
    feats: dict[str, float] = {f: np.nan for f in FEATURE_NAMES}

    bvp = _strip_nan(bvp_raw)
    eda = _strip_nan(eda_raw)
    if len(bvp) < int(2 * fs):
        return feats

    try:
        x = _bandpass_bvp(bvp, 0.5, 8.0, fs)
    except Exception:
        return feats
    if not np.isfinite(x).all() or np.std(x) <= 0:
        return feats

    seg_dur = len(x) / fs

    # Windowed max-abs envelope per second-bin
    try:
        centres, amps_win = _windowed_max_amp(x, fs, win_s=1.0)
        feats["bvp_amp_halfdecay_time_s"] = _decay_time(centres, amps_win, 0.5)
        feats["bvp_amp_quartdecay_time_s"] = _decay_time(centres, amps_win, 0.75)
    except Exception:
        pass

    # Per-beat amplitudes & timings
    peaks, beat_amps, peak_times = _per_beat_amplitudes(x, fs)

    # EDA-vs-BVP onset latency
    try:
        t_eda = _eda_first_slope_event_s(eda, fs)
        t_bvp_drop = _bvp_first_amp_drop_s(peak_times, beat_amps, drop_frac=0.15)
        if np.isfinite(t_eda) and np.isfinite(t_bvp_drop):
            # Positive = EDA first
            feats["bvp_onset_latency_vs_eda_s"] = float(t_bvp_drop - t_eda)
    except Exception:
        pass

    # Local vasoconstriction index: mean amp [0,3] - mean amp [7,10] / total mean
    try:
        a_early = _segment_window_amp(peak_times, beat_amps, 0.0, 3.0)
        a_late = _segment_window_amp(peak_times, beat_amps, 7.0, min(10.0, seg_dur))
        a_total = float(np.nanmean(beat_amps)) if len(beat_amps) else np.nan
        if np.isfinite(a_early) and np.isfinite(a_late) and np.isfinite(a_total) and abs(a_total) > EPS:
            feats["bvp_local_vasoconstriction_index"] = (a_early - a_late) / a_total
    except Exception:
        pass

    # Trend early vs late (first 3 s vs last 3 s)
    try:
        a_first = _segment_window_amp(peak_times, beat_amps, 0.0, 3.0)
        last_lo = max(0.0, seg_dur - 3.0)
        a_last = _segment_window_amp(peak_times, beat_amps, last_lo, seg_dur)
        if np.isfinite(a_first) and np.isfinite(a_last):
            feats["bvp_beat_amp_trend_diff"] = float(a_first - a_last)
            if abs(a_last) > EPS:
                feats["bvp_beat_amp_trend_ratio"] = float(a_first / a_last)
    except Exception:
        pass

    # Local vasoconstriction speed: slope (linear regression) of beat amp vs time
    try:
        finite = np.isfinite(beat_amps)
        if finite.sum() >= 4:
            t_fit = peak_times[finite]
            a_fit = beat_amps[finite]
            slope, intercept, r, p, _ = spstats.linregress(t_fit, a_fit)
            feats["bvp_local_vasoconstriction_speed_slope"] = float(slope)
            feats["bvp_local_vasoconstriction_speed_r2"] = float(r * r)
    except Exception:
        pass

    # Amplitude jitter (std)
    try:
        finite = np.isfinite(beat_amps)
        if finite.sum() >= 2:
            feats["bvp_amplitude_jitter"] = float(np.std(beat_amps[finite], ddof=1))
    except Exception:
        pass

    # CV of inter-beat intervals
    try:
        if len(peak_times) >= 3:
            rr = np.diff(peak_times)
            mu_rr = float(np.mean(rr))
            if mu_rr > 0:
                feats["bvp_cv_interbeat"] = float(np.std(rr, ddof=1) / mu_rr)
    except Exception:
        pass

    # Recovery halftime: from min-amp beat back to 50% of initial peak amplitude
    try:
        finite = np.isfinite(beat_amps)
        if finite.sum() >= 4:
            init_peak = float(np.nanmax(beat_amps[: max(1, len(beat_amps) // 4)]))
            min_idx = int(np.nanargmin(beat_amps))
            t_min = float(peak_times[min_idx])
            target = 0.5 * init_peak
            after = beat_amps[min_idx + 1 :]
            after_t = peak_times[min_idx + 1 :]
            hits = np.where(after >= target)[0]
            if len(hits) > 0:
                feats["bvp_recovery_halftime_s"] = float(after_t[hits[0]] - t_min)
            feats["bvp_local_min_amp_s"] = t_min
    except Exception:
        pass

    # Envelope slope + AUC over full segment
    try:
        analytic = spsig.hilbert(x - np.mean(x))
        env = np.abs(analytic)
        t_env = np.arange(len(env)) / fs
        slope, _, _, _, _ = spstats.linregress(t_env, env)
        feats["bvp_peak_envelope_slope"] = float(slope)
        feats["bvp_peak_envelope_auc"] = float(np.sum(np.abs(x)) / fs)
    except Exception:
        pass

    # First-derivative peak time
    try:
        d1 = np.diff(x) * fs
        if len(d1) > 1:
            idx = int(np.argmax(np.abs(d1)))
            feats["bvp_first_derivative_peak_time"] = float(idx / fs)
    except Exception:
        pass

    return feats


# ---------------------------------------------------------------------------
# Build feature matrix (both splits)
# ---------------------------------------------------------------------------
def _spo2_flatline_segments() -> set[str]:
    fp = TABLES_DIR / "inventory_constant_segments.csv"
    if not fp.exists():
        return set()
    df = pd.read_csv(fp)
    df = df[df["signal"] == "SpO2"]
    return set(df["segment_id"].astype(str).tolist())


def _spo2_is_flat_inline(spo2_raw: np.ndarray) -> bool:
    """Fallback flatline detector if a segment isn't on the inventory list."""
    s = _strip_nan(spo2_raw)
    if len(s) < 5:
        return True
    return float(np.std(s)) < 1e-6


def build_feature_matrix() -> pd.DataFrame:
    spo2_flat = _spo2_flatline_segments()
    rows: list[dict] = []
    for split in ("train", "validation"):
        print(f"[load] {split}")
        tensor, meta = load_split(split)
        n = len(meta)
        for i in tqdm(range(n), desc=f"feat {split}", ncols=88):
            r = meta.iloc[i]
            seg_id = str(r["segment_id"])
            base = {k: r[k] for k in META_COLS}
            spo2_bad = (seg_id in spo2_flat) or _spo2_is_flat_inline(tensor[i, SPO2_IDX])
            try:
                feats = extract_kinetics(
                    tensor[i, BVP_IDX], tensor[i, EDA_IDX], fs=SFREQ
                )
            except Exception:
                feats = {f: np.nan for f in FEATURE_NAMES}
            if spo2_bad:
                # Per spec: NaN features for SpO2 flatline segments. Even though
                # SpO2 isn't directly used here, flatline often co-occurs with
                # generally degraded signal quality and we want to be safe.
                feats = {k: np.nan for k in feats}
            base.update(feats)
            rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Delta-from-baseline (subject NoPain mean)
# ---------------------------------------------------------------------------
def add_delta_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feats = FEATURE_NAMES
    base = (
        df[df["class"] == "NoPain"]
        .groupby("subject")[feats]
        .mean()
    )
    # Align baseline back to every row by subject
    aligned = df[["subject"]].merge(
        base, left_on="subject", right_index=True, how="left"
    )[feats]
    delta = df[feats].to_numpy(float) - aligned.to_numpy(float)
    delta_cols = [f"{f}__delta" for f in feats]
    out = df.copy()
    for j, col in enumerate(delta_cols):
        out[col] = delta[:, j]
    return out, delta_cols


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = int(np.sum(x[:, None] > y[None, :]))
    lt = int(np.sum(x[:, None] < y[None, :]))
    return float(gt - lt) / (len(x) * len(y))


def paired_subject_means(df: pd.DataFrame, feature: str
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arm = (
        df[df["class"] == "PainArm"].groupby("subject", as_index=False)[feature].mean()
    )
    hand = (
        df[df["class"] == "PainHand"].groupby("subject", as_index=False)[feature].mean()
    )
    m = arm.merge(hand, on="subject", suffixes=("_arm", "_hand")).dropna()
    return (
        m[f"{feature}_arm"].to_numpy(dtype=float),
        m[f"{feature}_hand"].to_numpy(dtype=float),
        m["subject"].to_numpy(),
    )


def fit_lmm(df_tr: pd.DataFrame, feature: str) -> float:
    if not HAVE_LMM:
        return np.nan
    sub = df_tr[df_tr["class"].isin(["PainArm", "PainHand"])][
        ["subject", "class", feature]
    ].dropna()
    if sub["subject"].nunique() < 5 or sub[feature].std() == 0:
        return np.nan
    sub = sub.copy()
    sub["is_hand"] = (sub["class"] == "PainHand").astype(int)
    try:
        md = MixedLM.from_formula(
            f"Q('{feature}') ~ is_hand", groups="subject", data=sub
        )
        mdf = md.fit(method="lbfgs", reml=True, disp=False)
        return float(mdf.pvalues.get("is_hand", np.nan))
    except Exception:
        return np.nan


def run_armhand_tests(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    tr = df[df["split"] == "train"].copy()
    rows: list[dict] = []
    for feat in tqdm(features, desc="paired wilcoxon", ncols=88):
        x, y, _ = paired_subject_means(tr, feat)
        row: dict = {"feature": feat, "n_subjects": int(len(x))}
        try:
            if len(x) > 3 and np.any(x - y != 0):
                W, p = spstats.wilcoxon(x, y, zero_method="wilcox")
                row["wilcoxon_W"] = float(W)
                row["wilcoxon_p"] = float(p)
            else:
                row["wilcoxon_W"] = np.nan
                row["wilcoxon_p"] = np.nan
        except Exception:
            row["wilcoxon_W"] = np.nan
            row["wilcoxon_p"] = np.nan
        row["cliffs_delta"] = cliffs_delta(x, y)
        d = x - y
        d_nz = d[np.isfinite(d) & (d != 0)]
        if len(d_nz):
            n_arm_gt_hand = int((d_nz > 0).sum())
            row["n_pos_arm_gt_hand"] = n_arm_gt_hand
            row["n_neg_arm_lt_hand"] = int((d_nz < 0).sum())
            row["sign_consistency_arm_gt_hand"] = float(n_arm_gt_hand / len(d_nz))
        else:
            row["n_pos_arm_gt_hand"] = 0
            row["n_neg_arm_lt_hand"] = 0
            row["sign_consistency_arm_gt_hand"] = np.nan
        row["mean_arm"] = float(np.mean(x)) if len(x) else np.nan
        row["mean_hand"] = float(np.mean(y)) if len(y) else np.nan
        row["direction"] = (
            "ARM > HAND" if (row["mean_arm"] - row["mean_hand"]) > 0 else "HAND > ARM"
        )
        row["lmm_p"] = fit_lmm(tr, feat)
        rows.append(row)
    out = pd.DataFrame(rows)

    # BH-FDR across all features (raw + delta together)
    p = out["wilcoxon_p"].to_numpy()
    mask = np.isfinite(p)
    p_fdr = np.full_like(p, np.nan, dtype=float)
    if mask.sum() > 0:
        _, padj, _, _ = multipletests(p[mask], alpha=0.05, method="fdr_bh")
        p_fdr[mask] = padj
    out["wilcoxon_p_fdr"] = p_fdr

    # FDR for LMM as well
    p2 = out["lmm_p"].to_numpy()
    mask2 = np.isfinite(p2)
    p_fdr2 = np.full_like(p2, np.nan, dtype=float)
    if mask2.sum() > 0:
        _, padj2, _, _ = multipletests(p2[mask2], alpha=0.05, method="fdr_bh")
        p_fdr2[mask2] = padj2
    out["lmm_p_fdr"] = p_fdr2
    return out


# ---------------------------------------------------------------------------
# Validation reproducibility
# ---------------------------------------------------------------------------
def validation_reproducibility(df: pd.DataFrame, tests: pd.DataFrame
                                ) -> pd.DataFrame:
    nominal = tests[tests["wilcoxon_p"] < 0.05]["feature"].tolist()
    val = df[df["split"] == "validation"]
    tr = df[df["split"] == "train"]
    rows: list[dict] = []
    for feat in nominal:
        tr_arm, tr_hand, _ = paired_subject_means(tr, feat)
        va_arm, va_hand, _ = paired_subject_means(val, feat)
        tr_eff = float(np.mean(tr_arm) - np.mean(tr_hand)) if len(tr_arm) else np.nan
        va_eff = float(np.mean(va_arm) - np.mean(va_hand)) if len(va_arm) else np.nan
        preserved = (
            np.isfinite(tr_eff) and np.isfinite(va_eff)
            and np.sign(tr_eff) == np.sign(va_eff)
        )
        rows.append({
            "feature": feat,
            "train_arm_minus_hand": tr_eff,
            "val_arm_minus_hand": va_eff,
            "train_direction_preserved_in_val": bool(preserved),
            "val_n_subjects_paired": int(len(va_arm)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_top10_wilcoxon_bar(tests: pd.DataFrame) -> None:
    t = tests.dropna(subset=["wilcoxon_p"]).copy()
    t["abs_W"] = t["wilcoxon_W"].abs()
    t = t.sort_values("wilcoxon_p", ascending=True).head(10)
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#dd8452" if d == "ARM > HAND" else "#4c72b0" for d in t["direction"]]
    ax.barh(t["feature"][::-1], -np.log10(t["wilcoxon_p"][::-1].clip(lower=1e-12)),
            color=colors[::-1])
    ax.set_xlabel(r"-$\log_{10}(p)$ (paired Wilcoxon)")
    ax.set_title("Top 10 BVP-kinetics features (ARM vs HAND, train)")
    from matplotlib.patches import Patch
    legend = [Patch(color="#dd8452", label="ARM > HAND"),
              Patch(color="#4c72b0", label="HAND > ARM")]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top10_bvp_kinetics_wilcoxon_bar.png")
    plt.close(fig)


def plot_amp_halfdecay_violin(df: pd.DataFrame) -> None:
    feat = "bvp_amp_halfdecay_time_s"
    tr = df[(df["split"] == "train") & df["class"].isin(["PainArm", "PainHand"])]
    sub = tr[["class", feat]].dropna()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=sub, x="class", y=feat,
        palette={"PainArm": "#dd8452", "PainHand": "#c44e52"},
        inner="quartile", ax=ax,
    )
    ax.set_title("BVP amplitude halfdecay time by stimulation site (train)")
    ax.set_ylabel("halfdecay time (s)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "amp_halfdecay_violin.png")
    plt.close(fig)


def plot_vasoconstriction_index_hexbin(df: pd.DataFrame) -> None:
    fx = "bvp_local_vasoconstriction_index"
    fy = "bvp_amp_halfdecay_time_s"
    tr = df[(df["split"] == "train") & df["class"].isin(["PainArm", "PainHand"])]
    sub = tr[["class", fx, fy]].dropna()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cls, color in zip(axes, ["PainArm", "PainHand"], ["Oranges", "Reds"]):
        s = sub[sub["class"] == cls]
        if s.empty:
            continue
        hb = ax.hexbin(s[fx], s[fy], gridsize=25, cmap=color, mincnt=1)
        plt.colorbar(hb, ax=ax, label="count")
        ax.set_xlabel(fx)
        ax.set_title(f"{cls} (n={len(s)})")
    axes[0].set_ylabel(fy)
    fig.suptitle("Local vasoconstriction index vs halfdecay time")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "vasoconstriction_index_hexbin.png")
    plt.close(fig)


def plot_per_subject_effect_heatmap(df: pd.DataFrame, tests: pd.DataFrame) -> None:
    raw = tests[~tests["feature"].str.endswith("__delta")]
    top = (
        raw.dropna(subset=["wilcoxon_p"])
        .sort_values("wilcoxon_p")
        .head(10)["feature"]
        .tolist()
    )
    if not top:
        return
    tr = df[(df["split"] == "train") & df["class"].isin(["PainArm", "PainHand"])]
    sub_means = tr.groupby(["subject", "class"])[top].mean()
    pain_std = tr.groupby("subject")[top].std()
    subjects = sorted(tr["subject"].unique())
    mat = np.full((len(subjects), len(top)), np.nan)
    for i, sid in enumerate(subjects):
        try:
            ma = sub_means.loc[(sid, "PainArm")]
            mh = sub_means.loc[(sid, "PainHand")]
        except KeyError:
            continue
        std_s = pain_std.loc[sid] if sid in pain_std.index else None
        if std_s is None:
            continue
        diff = (ma - mh) / (std_s.replace(0, np.nan) + EPS)
        mat[i] = diff.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, max(6, len(subjects) * 0.22)))
    sns.heatmap(
        mat, cmap="RdBu_r", center=0,
        xticklabels=top, yticklabels=subjects,
        cbar_kws={"label": "(ARM-HAND)/std (per-subject z)"},
        ax=ax,
    )
    ax.set_xlabel("feature")
    ax.set_ylabel("subject")
    ax.set_title("Per-subject standardised ARM-HAND difference (top 10 features)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "per_subject_effect_heatmap.png")
    plt.close(fig)


def plot_raw_vs_delta_pvalue_scatter(tests: pd.DataFrame) -> None:
    raw = tests[~tests["feature"].str.endswith("__delta")].set_index("feature")
    delta = (
        tests[tests["feature"].str.endswith("__delta")]
        .assign(base=lambda d: d["feature"].str.replace("__delta", "", regex=False))
        .set_index("base")
    )
    common = raw.index.intersection(delta.index)
    if not len(common):
        return
    pr = raw.loc[common, "wilcoxon_p"].to_numpy(float)
    pd_ = delta.loc[common, "wilcoxon_p"].to_numpy(float)
    mask = np.isfinite(pr) & np.isfinite(pd_)
    if mask.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        -np.log10(pr[mask].clip(min=1e-12)),
        -np.log10(pd_[mask].clip(min=1e-12)),
        s=70, alpha=0.7, color="#4c72b0", edgecolor="k",
    )
    for i, name in enumerate(np.array(common)[mask]):
        ax.annotate(name, (
            -np.log10(max(pr[mask][i], 1e-12)),
            -np.log10(max(pd_[mask][i], 1e-12)),
        ), fontsize=7, alpha=0.8)
    lim = max(
        -np.log10(np.min(pr[mask])),
        -np.log10(np.min(pd_[mask])),
    ) + 0.5
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4)
    ax.set_xlabel(r"-$\log_{10} p$ (raw feature)")
    ax.set_ylabel(r"-$\log_{10} p$ ($\Delta$-from-baseline)")
    ax.set_title("Raw vs Delta p-values (above diagonal = baseline-subtraction helps)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "raw_vs_delta_pvalue_scatter.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(tests: pd.DataFrame, val_repro: pd.DataFrame,
                 n_train_subj: int, n_val_subj: int) -> None:
    n_fdr = int((tests["wilcoxon_p_fdr"] < 0.05).sum())
    n_nominal = int((tests["wilcoxon_p"] < 0.05).sum())
    fdr_rows = tests[tests["wilcoxon_p_fdr"] < 0.05].sort_values("wilcoxon_p")

    top10 = tests.dropna(subset=["wilcoxon_p"]).sort_values("wilcoxon_p").head(10)
    max_abs_cd = float(np.nanmax(np.abs(tests["cliffs_delta"]))) if len(tests) else np.nan
    median_abs_cd = float(np.nanmedian(np.abs(tests["cliffs_delta"]))) if len(tests) else np.nan

    val_n = int(len(val_repro))
    val_pres = int(val_repro["train_direction_preserved_in_val"].sum()) if val_n else 0

    # Hypothesis check: hand-stim should show
    #   * faster halfdecay (HAND < ARM) on bvp_amp_halfdecay_time_s
    #   * larger local_vasoconstriction_index (HAND > ARM)
    halfdecay = tests[tests["feature"] == "bvp_amp_halfdecay_time_s"]
    vasc = tests[tests["feature"] == "bvp_local_vasoconstriction_index"]

    def _fmt_dir(row: pd.DataFrame, expected: str) -> str:
        if row.empty:
            return "n/a"
        r = row.iloc[0]
        return (
            f"observed {r['direction']} (expected {expected}); "
            f"W={r['wilcoxon_W']:.2f}, p={r['wilcoxon_p']:.3g}, "
            f"Cliff={r['cliffs_delta']:+.3f}"
        )

    halfdecay_str = _fmt_dir(halfdecay, "HAND < ARM")
    vasc_str = _fmt_dir(vasc, "HAND > ARM")

    lines: list[str] = []
    lines.append("# 14 - Tier-A2: BVP proximity kinetics (ARM vs HAND)")
    lines.append("")
    lines.append("## Headline")
    lines.append(
        f"- Features tested: **{len(tests)}** (raw + Delta-from-baseline). "
        f"Train subjects: {n_train_subj}, validation subjects: {n_val_subj}."
    )
    lines.append(
        f"- **{n_fdr}** features survive BH-FDR<0.05; **{n_nominal}** at "
        f"nominal p<0.05."
    )
    lines.append(
        f"- |Cliff's delta| range: max={max_abs_cd:.3f}, median={median_abs_cd:.3f} "
        f"(prior aggregate ceiling was ~0.14-0.18)."
    )
    lines.append("")
    lines.append("## Top 10 features by smallest p (paired Wilcoxon, train)")
    lines.append("")
    lines.append("| feature | n | W | p | p_fdr | Cliff | sign(arm>hand) | direction | LMM p |")
    lines.append("|---------|--:|--:|--:|------:|------:|---------------:|-----------|------:|")
    for _, r in top10.iterrows():
        lines.append(
            f"| `{r['feature']}` | {int(r['n_subjects'])} | "
            f"{r['wilcoxon_W']:.1f} | {r['wilcoxon_p']:.3g} | "
            f"{r['wilcoxon_p_fdr']:.3g} | {r['cliffs_delta']:+.3f} | "
            f"{r['sign_consistency_arm_gt_hand']:.2f} | {r['direction']} | "
            f"{r['lmm_p']:.3g} |"
        )
    lines.append("")
    lines.append("## Hypothesis-driven probes")
    lines.append(f"- `bvp_amp_halfdecay_time_s`: {halfdecay_str}.")
    lines.append(f"- `bvp_local_vasoconstriction_index`: {vasc_str}.")
    lines.append("")
    if not halfdecay.empty and not vasc.empty:
        h_ok = halfdecay.iloc[0]["direction"] == "HAND > ARM"  # NOTE: HAND > ARM means ARM is shorter; we want HAND shorter
        # Hand should decay faster -> halfdecay smaller -> direction HAND < ARM -> direction string would be "ARM > HAND"
        h_ok = halfdecay.iloc[0]["direction"] == "ARM > HAND"
        v_ok = vasc.iloc[0]["direction"] == "HAND > ARM"
        verdict = "Yes" if (h_ok and v_ok) else "Partially" if (h_ok or v_ok) else "No"
        lines.append(
            f"- **Hypothesis verdict (proximity)**: {verdict}. "
            f"Halfdecay direction matches hand-faster-decay = {h_ok}; "
            f"local-vasoconstriction-index hand-larger = {v_ok}."
        )
        lines.append("")

    lines.append("## Validation reproducibility")
    lines.append(
        f"- Features at nominal p<0.05 in train: **{val_n}**. "
        f"Direction preserved on validation: **{val_pres}/{val_n}** "
        f"({(100 * val_pres / max(val_n, 1)):.1f}%)."
    )
    lines.append("")

    lines.append("## Effect size vs prior 0.14-0.18 ceiling")
    if max_abs_cd > 0.18:
        lines.append(
            f"- **Beats the ceiling**: max |Cliff's delta| = {max_abs_cd:.3f} > 0.18."
        )
    else:
        lines.append(
            f"- Does **not** decisively beat the ceiling (max |Cliff's delta| = "
            f"{max_abs_cd:.3f} <= 0.18)."
        )
    lines.append("")

    lines.append("## Outputs")
    lines.append("- `results/tables/tierA2_bvp_kinetics_features.parquet`")
    lines.append("- `results/tables/tierA2_bvp_kinetics_tests.csv`")
    lines.append("- `results/tables/tierA2_bvp_kinetics_val_repro.csv`")
    lines.append("- `plots/tierA2_bvp_kinetics/`")

    out_fp = REPORTS_DIR / "14_tierA2_bvp_kinetics_summary.md"
    out_fp.write_text("\n".join(lines))
    print(f"[save] {out_fp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[1/5] Building BVP-kinetics features ...")
    feats_df = build_feature_matrix()
    feats_df, delta_cols = add_delta_features(feats_df)
    all_feats = FEATURE_NAMES + delta_cols
    parquet_fp = TABLES_DIR / "tierA2_bvp_kinetics_features.parquet"
    feats_df.to_parquet(parquet_fp, index=False)
    print(
        f"  saved {parquet_fp} ({len(feats_df)} segments x "
        f"{len(all_feats)} features)"
    )

    print("[2/5] Paired Wilcoxon ARM vs HAND (subject-means, train) ...")
    tests = run_armhand_tests(feats_df, all_feats)
    tests_fp = TABLES_DIR / "tierA2_bvp_kinetics_tests.csv"
    tests.to_csv(tests_fp, index=False)
    print(f"  saved {tests_fp}; "
          f"FDR<0.05: {(tests['wilcoxon_p_fdr'] < 0.05).sum()}")

    print("[3/5] Validation-split direction preservation ...")
    val_repro = validation_reproducibility(feats_df, tests)
    val_fp = TABLES_DIR / "tierA2_bvp_kinetics_val_repro.csv"
    val_repro.to_csv(val_fp, index=False)
    print(f"  saved {val_fp} ({len(val_repro)} nominal-sig features)")

    print("[4/5] Plots ...")
    try:
        plot_top10_wilcoxon_bar(tests)
    except Exception as e:
        print(f"  top10 bar failed: {e}")
    try:
        plot_amp_halfdecay_violin(feats_df)
    except Exception as e:
        print(f"  halfdecay violin failed: {e}")
    try:
        plot_vasoconstriction_index_hexbin(feats_df)
    except Exception as e:
        print(f"  hexbin failed: {e}")
    try:
        plot_per_subject_effect_heatmap(feats_df, tests)
    except Exception as e:
        print(f"  per-subject heatmap failed: {e}")
    try:
        plot_raw_vs_delta_pvalue_scatter(tests)
    except Exception as e:
        print(f"  raw vs delta scatter failed: {e}")

    print("[5/5] Writing report ...")
    n_train_subj = int(feats_df[feats_df["split"] == "train"]["subject"].nunique())
    n_val_subj = int(feats_df[feats_df["split"] == "validation"]["subject"].nunique())
    write_report(tests, val_repro, n_train_subj, n_val_subj)
    print("Done.")


if __name__ == "__main__":
    main()
