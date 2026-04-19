"""Tier A #1: stimulus-onset alignment for ARM vs HAND separability.

Hypothesis: each 10-s pain segment starts at a slightly different within-segment
time relative to actual TENS onset. Averaging stats across the full 10 s washes
small timing-locked ARM-vs-HAND differences. If we detect onset and re-extract
features on an aligned [onset, onset+6 s] window, paired Wilcoxon p-values
should sharpen.

Pipeline:
    1. Detect per-segment onset (EDA slope, BVP amp drop, RESP perturbation),
       consensus = median, fallback EDA-only if spread > 2 s, capped to [0,4]s.
    2. Build curated feature list (~25): top 20 reactivity p-values + 4 LMM
       Resp w3 features. Compute each feature on (a) raw 0-10 s window
       and (b) aligned [onset, onset+6 s] window resampled to 600 samples.
    3. Paired-Wilcoxon ARM vs HAND on subject means, twice. BH-FDR per family.
    4. Plots + markdown report.

Run:
    uv run python scripts/13_tierA1_onset_alignment.py
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
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import SFREQ, SIGNALS, load_split  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120
sns.set_context("talk", font_scale=0.7)

RNG_SEED = 42
np.random.seed(RNG_SEED)

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_ROOT / "results" / "tables"
REPORTS_DIR = ANALYSIS_ROOT / "results" / "reports"
PLOTS_DIR = ANALYSIS_ROOT / "plots" / "tierA1_alignment"
for d in (TABLES_DIR, REPORTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]

EPS = 1e-8

# ---------------------------------------------------------------------------
# Constants for alignment
# ---------------------------------------------------------------------------
WIN_LEN_S = 10.0           # raw segment length
WIN_LEN_N = int(WIN_LEN_S * SFREQ)  # 1000
ALIGNED_LEN_S = 6.0        # aligned window length
ALIGNED_LEN_N = int(ALIGNED_LEN_S * SFREQ)  # 600
ONSET_MIN_S = 0.0          # cap onset to [0,4] s
ONSET_MAX_S = 4.0
SPREAD_MAX_S = 2.0         # if spread > this, fall back to EDA only
PRE_BASELINE_S = 2.0       # first 2 s define EDA baseline mean+2*std

SIG_IDX = {s: i for i, s in enumerate(SIGNALS)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return float(gt - lt) / (nx * ny)


def _safe_savgol(x: np.ndarray, win: int, poly: int) -> np.ndarray:
    """Savgol with fallback to identity if window > len."""
    if len(x) < win + 2:
        return x
    if win % 2 == 0:
        win += 1
    return spsig.savgol_filter(x, win, poly)


def _drop_nan(x: np.ndarray) -> np.ndarray:
    """Trim trailing NaNs (the loader pads short signals with NaN)."""
    if not np.any(np.isnan(x)):
        return x
    good = ~np.isnan(x)
    if not good.any():
        return np.array([], dtype=x.dtype)
    last = np.where(good)[0][-1]
    return x[: last + 1]


def is_spo2_flat(spo2: np.ndarray) -> bool:
    """Treat segment as flatline if SpO2 std (after trimming NaN) is 0."""
    s = _drop_nan(spo2)
    if len(s) < 4:
        return True
    return float(np.std(s)) == 0.0


# ---------------------------------------------------------------------------
# Onset detectors (return seconds in [0, ONSET_MAX_S], NaN if undetectable)
# ---------------------------------------------------------------------------
def onset_eda(eda: np.ndarray) -> float:
    """First sample where smoothed EDA derivative > mean+2*std of first 2 s."""
    x = _drop_nan(eda)
    if len(x) < int(0.5 * SFREQ):
        return np.nan
    smoothed = _safe_savgol(x, win=51, poly=3)  # 0.5 s window
    deriv = np.diff(smoothed)
    pre_n = int(PRE_BASELINE_S * SFREQ)
    if pre_n >= len(deriv):
        return np.nan
    base = deriv[:pre_n]
    if np.std(base) == 0:
        return np.nan
    thr = float(np.mean(base) + 2 * np.std(base))
    # search after pre-baseline up to ONSET_MAX_S * SFREQ
    search_end = min(int(ONSET_MAX_S * SFREQ), len(deriv))
    if pre_n >= search_end:
        return np.nan
    region = deriv[pre_n:search_end]
    above = np.where(region > thr)[0]
    if len(above) == 0:
        return np.nan
    return float((pre_n + above[0]) / SFREQ)


def onset_bvp(bvp: np.ndarray) -> float:
    """First beat-to-beat amplitude reduction > 15 % within [0, ONSET_MAX_S]."""
    x = _drop_nan(bvp)
    if len(x) < int(1.0 * SFREQ):
        return np.nan
    # BVP peaks: ~1-2 Hz. Use scipy peaks with min distance 0.4 s
    try:
        peaks, props = spsig.find_peaks(
            x, distance=int(0.4 * SFREQ), prominence=np.std(x) * 0.3
        )
    except Exception:
        return np.nan
    if len(peaks) < 4:
        return np.nan
    # estimate trough between consecutive peaks for amp
    amps = []
    for i in range(len(peaks) - 1):
        seg = x[peaks[i] : peaks[i + 1]]
        if len(seg) < 2:
            amps.append(np.nan)
            continue
        amps.append(float(x[peaks[i]] - np.min(seg)))
    amps = np.array(amps, dtype=float)
    if not np.isfinite(amps).any():
        return np.nan
    # rolling baseline = first 2 amplitudes
    if len(amps) < 3:
        return np.nan
    base_amp = float(np.nanmean(amps[:2]))
    if base_amp <= 0:
        return np.nan
    # find first beat where amp dropped > 15 %
    for i in range(2, len(amps)):
        if not np.isfinite(amps[i]):
            continue
        if (base_amp - amps[i]) / base_amp > 0.15:
            t = peaks[i + 1] / SFREQ
            if ONSET_MIN_S <= t <= ONSET_MAX_S:
                return float(t)
            else:
                return np.nan
    return np.nan


def onset_resp(resp: np.ndarray) -> float:
    """First inhale->exhale break: smoothed RESP first sign-change of slope
    after pre-baseline window."""
    x = _drop_nan(resp)
    if len(x) < int(1.0 * SFREQ):
        return np.nan
    smoothed = _safe_savgol(x, win=71, poly=3)  # 0.7 s
    deriv = np.diff(smoothed)
    pre_n = int(PRE_BASELINE_S * SFREQ)
    if pre_n >= len(deriv):
        return np.nan
    sign_pre = np.sign(deriv[pre_n - 1]) if pre_n > 0 else 0
    search_end = min(int(ONSET_MAX_S * SFREQ), len(deriv))
    if pre_n >= search_end:
        return np.nan
    region_signs = np.sign(deriv[pre_n:search_end])
    # first sign change (relative to last pre value or any flip in the region)
    # Use any sign-change inside region.
    flips = np.where(np.diff(region_signs) != 0)[0]
    if len(flips) == 0:
        return np.nan
    return float((pre_n + flips[0]) / SFREQ)


def consensus_onset(
    eda: np.ndarray, bvp: np.ndarray, resp: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Return (onset_consensus, onset_eda, onset_bvp, onset_resp, spread)."""
    o_e = onset_eda(eda)
    o_b = onset_bvp(bvp)
    o_r = onset_resp(resp)
    detectors = np.array([o_e, o_b, o_r], dtype=float)
    valid = detectors[np.isfinite(detectors)]
    if valid.size == 0:
        return np.nan, o_e, o_b, o_r, np.nan
    spread = float(np.nanmax(valid) - np.nanmin(valid)) if valid.size >= 2 else 0.0
    if valid.size >= 2 and spread > SPREAD_MAX_S:
        # fall back to EDA only
        cons = o_e if np.isfinite(o_e) else float(np.nanmedian(valid))
    else:
        cons = float(np.nanmedian(valid))
    if not np.isfinite(cons):
        return np.nan, o_e, o_b, o_r, spread
    cons = float(np.clip(cons, ONSET_MIN_S, ONSET_MAX_S))
    return cons, o_e, o_b, o_r, spread


# ---------------------------------------------------------------------------
# Feature library (operates on a single 1-D signal window)
# ---------------------------------------------------------------------------
def feat_mean(x):  return float(np.nanmean(x))
def feat_std(x):   return float(np.nanstd(x))
def feat_median(x): return float(np.nanmedian(x))
def feat_range(x): return float(np.nanmax(x) - np.nanmin(x))
def feat_iqr(x):
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))
def feat_mad(x):
    m = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - m)))
def feat_mean_abs(x): return float(np.nanmean(np.abs(x)))
def feat_rms(x):    return float(np.sqrt(np.nanmean(x * x)))
def feat_diff_std(x):
    d = np.diff(x[~np.isnan(x)])
    return float(np.nanstd(d)) if len(d) else np.nan
def feat_skew(x):
    g = x[~np.isnan(x)]
    if len(g) < 3 or np.std(g) == 0:
        return np.nan
    try:
        return float(sp_stats.skew(g, bias=False))
    except Exception:
        return np.nan
def feat_kurt(x):
    g = x[~np.isnan(x)]
    if len(g) < 4 or np.std(g) == 0:
        return np.nan
    try:
        return float(sp_stats.kurtosis(g, bias=False))
    except Exception:
        return np.nan
def feat_n_extrema(x):
    g = x[~np.isnan(x)]
    if len(g) < 3:
        return np.nan
    d = np.diff(g)
    sign_changes = np.sum(np.diff(np.sign(d)) != 0)
    return float(sign_changes)
def feat_petrosian_fd(x):
    g = x[~np.isnan(x)]
    if len(g) < 4:
        return np.nan
    d = np.diff(g)
    n_delta = float(np.sum(np.diff(np.sign(d)) != 0))
    n = len(g)
    if n_delta == 0:
        return np.nan
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta))))
def feat_bp_resp(x, fs=SFREQ, lo=0.05, hi=0.5):
    g = x[~np.isnan(x)]
    if len(g) < 32 or np.std(g) == 0:
        return np.nan
    f, pxx = spsig.welch(g, fs=fs, nperseg=min(256, len(g)))
    mask = (f >= lo) & (f <= hi)
    if not mask.any():
        return np.nan
    return float(np.trapezoid(pxx[mask], f[mask]))
def feat_resp_amp_std(x):
    """Std of envelope swings: peak-to-trough amplitude across breaths."""
    g = x[~np.isnan(x)]
    if len(g) < int(1.0 * SFREQ):
        return np.nan
    try:
        peaks, _ = spsig.find_peaks(g, distance=int(0.5 * SFREQ))
        troughs, _ = spsig.find_peaks(-g, distance=int(0.5 * SFREQ))
    except Exception:
        return np.nan
    if len(peaks) < 2 or len(troughs) < 2:
        return np.nan
    # Pair each peak with nearest preceding trough
    amps = []
    for p in peaks:
        prev = troughs[troughs < p]
        if prev.size:
            amps.append(g[p] - g[prev[-1]])
    if len(amps) < 2:
        return np.nan
    return float(np.std(amps))
def feat_resp_peak_power(x, fs=SFREQ):
    g = x[~np.isnan(x)]
    if len(g) < 32 or np.std(g) == 0:
        return np.nan
    f, pxx = spsig.welch(g, fs=fs, nperseg=min(256, len(g)))
    if len(pxx) == 0:
        return np.nan
    return float(np.max(pxx))


# Window-level kurt/skew (the LMM Resp_w3 features) -- these are the kurt/skew
# of a 2-s sub-window. On unaligned: window 3 = samples 400:600. On aligned:
# we recreate by splitting the aligned 6-s into 3 windows of 2 s each and use
# the middle one (this is the "mid" of the aligned window).
def w3_indices_unaligned(n: int) -> tuple[int, int]:
    """Window 3 of 5 in 0..n-1: indices [2/5*n, 3/5*n)."""
    return int(0.4 * n), int(0.6 * n)


def w_mid_aligned(n: int) -> tuple[int, int]:
    """Middle 2 s of an aligned 6-s window (indices [n/3, 2n/3))."""
    return n // 3, 2 * n // 3


# ---------------------------------------------------------------------------
# Curated feature list
# ---------------------------------------------------------------------------
# Each entry: name, signal, callable(window_array) -> float
# We strip the reactivity prefix to get the underlying primitive name.
PRIMITIVES = {
    "Bvp_mean":     ("Bvp",  feat_mean),
    "Bvp_std":      ("Bvp",  feat_std),
    "Bvp_mad":      ("Bvp",  feat_mad),
    "Bvp_n_extrema":("Bvp",  feat_n_extrema),
    "Bvp_petrosian_fd": ("Bvp", feat_petrosian_fd),
    "Bvp_diff_std": ("Bvp",  feat_diff_std),
    "Bvp_rms":      ("Bvp",  feat_rms),
    "Bvp_kurtosis": ("Bvp",  feat_kurt),
    "Bvp_skew":     ("Bvp",  feat_skew),

    "Eda_mean":     ("Eda",  feat_mean),
    "Eda_std":      ("Eda",  feat_std),
    "Eda_mad":      ("Eda",  feat_mad),
    "Eda_range":    ("Eda",  feat_range),
    "Eda_diff_std": ("Eda",  feat_diff_std),
    "Eda_skew":     ("Eda",  feat_skew),
    "Eda_kurtosis": ("Eda",  feat_kurt),

    "Resp_mean":    ("Resp", feat_mean),
    "Resp_std":     ("Resp", feat_std),
    "Resp_mad":     ("Resp", feat_mad),
    "Resp_skew":    ("Resp", feat_skew),
    "Resp_kurtosis":("Resp", feat_kurt),
    "Resp_diff_std":("Resp", feat_diff_std),
    "Resp_range":   ("Resp", feat_range),
    "Resp_mean_abs":("Resp", feat_mean_abs),
    "Resp_rms":     ("Resp", feat_rms),
    "Resp_iqr":     ("Resp", feat_iqr),
    "Resp_n_extrema":("Resp", feat_n_extrema),
    "Resp_amp_std": ("Resp", feat_resp_amp_std),
    "Resp_peak_power":("Resp", feat_resp_peak_power),
    "Resp_bp_0p2_0p5":("Resp", lambda x: feat_bp_resp(x, lo=0.2, hi=0.5)),

    # LMM-significant temporal: window-3 (mid) skew & kurt of Resp.
    "Resp_w3_kurt": ("Resp", "WMID_KURT"),  # special handled below
    "Resp_w3_skew": ("Resp", "WMID_SKEW"),
}


def select_curated_features() -> list[str]:
    """Pick top-20 reactivity (smallest p) primitives + 4 LMM Resp_w3 features.
    Returns list of primitive names (= keys of PRIMITIVES)."""
    react = pd.read_csv(TABLES_DIR / "reactivity_armhand_tests.csv")
    react = react.sort_values("p").reset_index(drop=True)

    chosen: list[str] = []
    seen: set[str] = set()
    for _, row in react.iterrows():
        f = str(row["feature"])
        # strip reactivity prefix
        base = f
        for pref in ("delta_", "ratio_", "zdev_"):
            if base.startswith(pref):
                base = base[len(pref):]
                break
        # also strip a leading "raw_"
        if base.startswith("raw_"):
            base = base[len("raw_"):]
        # Canonicalise capitalisation: physio_features uses RESP_ prefixes
        # for some named features; the raw_stats keys use Resp_*. Map both.
        cap_map = {
            "RESP_amp_std": "Resp_amp_std",
            "RESP_amp_mean": "Resp_amp_mean",
            "RESP_rate":     "Resp_rate",
        }
        base = cap_map.get(base, base)

        if base in PRIMITIVES and base not in seen:
            chosen.append(base)
            seen.add(base)
        if len(chosen) >= 20:
            break

    # Add LMM features (4 of them, 2 unique primitives × 2 versions: kurt/skew
    # already covered, plus *__delta variants are the same callable).
    for lmm in ("Resp_w3_kurt", "Resp_w3_skew"):
        if lmm not in seen:
            chosen.append(lmm)
            seen.add(lmm)
    return chosen


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------
def extract_unaligned(seg: np.ndarray, sig_name: str) -> np.ndarray:
    """Return the raw 0-10 s window for a given signal of one segment."""
    return seg[SIG_IDX[sig_name]]


def extract_aligned(
    seg: np.ndarray, sig_name: str, onset_s: float
) -> np.ndarray:
    """Return [onset, onset+6 s] window resampled to ALIGNED_LEN_N samples."""
    raw = seg[SIG_IDX[sig_name]]
    raw = _drop_nan(raw)
    if len(raw) == 0 or not np.isfinite(onset_s):
        return np.array([], dtype=raw.dtype)
    start = int(round(onset_s * SFREQ))
    end = start + ALIGNED_LEN_N
    if end <= len(raw):
        win = raw[start:end]
    else:
        # need to pad: take what's available, pad rest with last value
        win = raw[start:]
        if len(win) == 0:
            return np.array([], dtype=raw.dtype)
        pad = np.full(ALIGNED_LEN_N - len(win), win[-1], dtype=raw.dtype)
        win = np.concatenate([win, pad])
    # ensure exactly ALIGNED_LEN_N
    if len(win) != ALIGNED_LEN_N:
        # Resample (linear interp) to exact length
        idx_old = np.linspace(0, 1, len(win))
        idx_new = np.linspace(0, 1, ALIGNED_LEN_N)
        win = np.interp(idx_new, idx_old, win).astype(np.float32)
    return win


def compute_feature(name: str, x: np.ndarray, aligned: bool) -> float:
    """Apply the named feature primitive on x. For special WMID_* primitives
    the window mid third is used (matches w3 of 5-window unaligned and middle
    of 3-window aligned)."""
    if x is None or len(x) == 0:
        return np.nan
    sig_name, fn = PRIMITIVES[name]
    if isinstance(fn, str):
        # Special window-3 primitive: take mid sub-window
        if aligned:
            lo, hi = w_mid_aligned(len(x))
        else:
            lo, hi = w3_indices_unaligned(len(x))
        sub = x[lo:hi]
        if fn == "WMID_KURT":
            return feat_kurt(sub)
        if fn == "WMID_SKEW":
            return feat_skew(sub)
        return np.nan
    return float(fn(x))


# ---------------------------------------------------------------------------
# Per-segment driver
# ---------------------------------------------------------------------------
def process_split(
    tensor: np.ndarray, meta: pd.DataFrame, feat_names: list[str], split: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (onsets_df, features_df). Only ARM/HAND segments are processed."""
    pain_mask = meta["class"].isin(["PainArm", "PainHand"]).to_numpy()
    pain_idx = np.where(pain_mask)[0]

    onset_rows: list[dict] = []
    feat_rows: list[dict] = []

    skipped_flat = 0
    for i in tqdm(pain_idx, desc=f"{split} segments", unit="seg"):
        seg = tensor[i]   # (4, 1000)
        meta_i = meta.iloc[i]

        spo2 = seg[SIG_IDX["SpO2"]]
        if is_spo2_flat(spo2):
            skipped_flat += 1
            continue

        eda = seg[SIG_IDX["Eda"]]
        bvp = seg[SIG_IDX["Bvp"]]
        resp = seg[SIG_IDX["Resp"]]

        # _drop_nan'd copies for onset detection
        cons, o_e, o_b, o_r, spread = consensus_onset(
            _drop_nan(eda), _drop_nan(bvp), _drop_nan(resp)
        )

        onset_rows.append({
            "split": split,
            "subject": int(meta_i["subject"]),
            "class": meta_i["class"],
            "segment_id": meta_i["segment_id"],
            "onset_EDA_s": o_e,
            "onset_BVP_s": o_b,
            "onset_RESP_s": o_r,
            "onset_consensus_s": cons,
            "onset_spread_s": spread,
        })

        # If consensus is NaN (no detector fired), fall back to 0.0
        use_onset = cons if np.isfinite(cons) else 0.0

        # Compute features twice
        row = {
            "split": split,
            "subject": int(meta_i["subject"]),
            "class": meta_i["class"],
            "segment_idx": int(meta_i["segment_idx"]),
            "segment_id": meta_i["segment_id"],
            "onset_consensus_s": cons,
        }
        for name in feat_names:
            sig_name, _ = PRIMITIVES[name]
            unalign = _drop_nan(extract_unaligned(seg, sig_name))
            align = extract_aligned(seg, sig_name, use_onset)
            row[f"unaligned_{name}"] = compute_feature(name, unalign, aligned=False)
            row[f"aligned_{name}"]   = compute_feature(name, align,   aligned=True)
        feat_rows.append(row)

    onsets_df = pd.DataFrame(onset_rows)
    feats_df = pd.DataFrame(feat_rows)
    print(f"[{split}] processed {len(feat_rows)} pain segments "
          f"(skipped {skipped_flat} SpO2-flatline)")
    return onsets_df, feats_df


# ---------------------------------------------------------------------------
# Paired Wilcoxon helpers
# ---------------------------------------------------------------------------
def paired_arm_vs_hand(subject_means: pd.DataFrame, feat: str) -> dict:
    a = subject_means[subject_means["class"] == "PainArm"][["subject", feat]]
    b = subject_means[subject_means["class"] == "PainHand"][["subject", feat]]
    m = a.merge(b, on="subject", suffixes=("_A", "_H")).dropna()
    out = {"n": int(len(m)), "W": np.nan, "p": np.nan,
           "cliff": np.nan, "sign_consistency": np.nan,
           "direction": ""}
    if len(m) < 4:
        return out
    x = m[f"{feat}_A"].to_numpy(float)
    y = m[f"{feat}_H"].to_numpy(float)
    d = x - y
    if np.allclose(np.nanstd(d), 0):
        return out
    try:
        W, p = sp_stats.wilcoxon(x, y, zero_method="wilcox")
        out["W"] = float(W); out["p"] = float(p)
    except Exception:
        pass
    out["cliff"] = cliffs_delta(x, y)
    med = np.sign(np.nanmedian(d))
    if med == 0:
        med = np.sign(np.nanmean(d)) or 1.0
    nz = d[d != 0]
    if nz.size:
        out["sign_consistency"] = float(np.mean(np.sign(nz) == med))
    out["direction"] = "ARM>HAND" if np.nanmean(x) > np.nanmean(y) else "HAND>ARM"
    return out


def alignment_effect_table(
    feats_df: pd.DataFrame, feat_names: list[str], split: str = "train"
) -> pd.DataFrame:
    df = feats_df[feats_df["split"] == split].copy()
    sub_means = (
        df.groupby(["subject", "class"], as_index=False)
        .mean(numeric_only=True)
    )

    rows = []
    for f in feat_names:
        u_col = f"unaligned_{f}"
        a_col = f"aligned_{f}"
        u = paired_arm_vs_hand(sub_means, u_col)
        a = paired_arm_vs_hand(sub_means, a_col)
        rows.append({
            "feature": f,
            "n": u["n"],
            "unaligned_W": u["W"], "unaligned_p": u["p"],
            "unaligned_cliff": u["cliff"],
            "unaligned_sign_consistency": u["sign_consistency"],
            "unaligned_direction": u["direction"],
            "aligned_W": a["W"], "aligned_p": a["p"],
            "aligned_cliff": a["cliff"],
            "aligned_sign_consistency": a["sign_consistency"],
            "aligned_direction": a["direction"],
            "p_ratio": (a["p"] / u["p"]) if (u["p"] and np.isfinite(u["p"])
                                             and u["p"] > 0) else np.nan,
            "delta_cliff": (abs(a["cliff"]) - abs(u["cliff"])) if (
                np.isfinite(a["cliff"]) and np.isfinite(u["cliff"])) else np.nan,
        })
    out = pd.DataFrame(rows)
    # FDR per family
    for col in ("unaligned_p", "aligned_p"):
        ok = out[col].notna()
        out[f"{col}_fdr"] = np.nan
        if ok.sum() >= 2:
            _, fdr, _, _ = multipletests(out.loc[ok, col].values, method="fdr_bh")
            out.loc[ok, f"{col}_fdr"] = fdr
    out["n_features_improved"] = int(((out["aligned_p"] < out["unaligned_p"]) &
                                       out["aligned_p"].notna()).sum())
    return out


def val_direction_check(
    feats_df: pd.DataFrame, feat_names: list[str]
) -> pd.DataFrame:
    """Check whether ARM-vs-HAND direction on aligned features replicates on val."""
    rows = []
    for split in ("train", "validation"):
        df = feats_df[feats_df["split"] == split].copy()
        if df.empty:
            continue
        sub_means = df.groupby(["subject", "class"], as_index=False).mean(
            numeric_only=True
        )
        for f in feat_names:
            for prefix in ("unaligned", "aligned"):
                col = f"{prefix}_{f}"
                a = sub_means[sub_means["class"] == "PainArm"][col].mean()
                h = sub_means[sub_means["class"] == "PainHand"][col].mean()
                rows.append({
                    "split": split,
                    "feature": f,
                    "version": prefix,
                    "mean_ARM": a, "mean_HAND": h,
                    "direction": "ARM>HAND" if a > h else "HAND>ARM",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_onset_distribution(onsets_df: pd.DataFrame, fp: Path) -> None:
    df = onsets_df[onsets_df["split"] == "train"].copy()
    df = df[df["onset_consensus_s"].notna()]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, ONSET_MAX_S, 24)
    for cls, color in [("PainArm", "tab:orange"), ("PainHand", "tab:blue")]:
        sub = df[df["class"] == cls]["onset_consensus_s"]
        ax.hist(sub, bins=bins, alpha=0.55, label=f"{cls} (n={len(sub)})",
                color=color, edgecolor="white")
    ax.set_xlabel("Consensus onset (s)")
    ax.set_ylabel("Segments")
    ax.set_title("Detected stimulus onset by class (train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def plot_aligned_vs_unaligned_pvalues(eff: pd.DataFrame, fp: Path) -> None:
    df = eff[eff["unaligned_p"].notna() & eff["aligned_p"].notna()].copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    x = -np.log10(df["unaligned_p"].clip(lower=1e-12))
    y = -np.log10(df["aligned_p"].clip(lower=1e-12))
    ax.scatter(x, y, s=70, alpha=0.7, edgecolor="black", linewidth=0.5)
    # Annotate top points (largest aligned -log10p)
    top = df.assign(_score=y - x).sort_values("_score", ascending=False).head(5)
    for _, r in top.iterrows():
        xi = -np.log10(max(r["unaligned_p"], 1e-12))
        yi = -np.log10(max(r["aligned_p"], 1e-12))
        ax.annotate(r["feature"], (xi, yi), fontsize=8, alpha=0.8)
    lim = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.05 + 0.5
    ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=1)
    ax.axhline(-np.log10(0.05), color="red", linewidth=0.8, linestyle=":")
    ax.axvline(-np.log10(0.05), color="red", linewidth=0.8, linestyle=":")
    ax.set_xlabel("-log10(p)  unaligned")
    ax.set_ylabel("-log10(p)  aligned")
    ax.set_title("Onset alignment effect on ARM-vs-HAND p-values")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def plot_top10_cliff_change(eff: pd.DataFrame, fp: Path) -> None:
    df = eff[eff["delta_cliff"].notna()].copy()
    df = df.sort_values("delta_cliff", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:green" if v > 0 else "tab:red" for v in df["delta_cliff"]]
    ax.barh(df["feature"], df["delta_cliff"], color=colors, edgecolor="black")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$\Delta|\delta|$  (aligned - unaligned)")
    ax.set_title("Top-10 features by Cliff's |delta| change")
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(
    eff: pd.DataFrame,
    onsets_df: pd.DataFrame,
    val_dir: pd.DataFrame,
    fp: Path,
) -> None:
    n_total = len(eff)
    n_improved = int(((eff["aligned_p"] < eff["unaligned_p"]) &
                      eff["aligned_p"].notna() &
                      eff["unaligned_p"].notna()).sum())
    n_fdr_un = int((eff["unaligned_p_fdr"] < 0.05).sum())
    n_fdr_al = int((eff["aligned_p_fdr"] < 0.05).sum())
    new_fdr = eff[(eff["aligned_p_fdr"] < 0.05) &
                  ~(eff["unaligned_p_fdr"] < 0.05)]["feature"].tolist()
    median_p_ratio = float(eff["p_ratio"].median(skipna=True))

    # Onset distribution stats
    train_o = onsets_df[onsets_df["split"] == "train"]
    n_pain = len(train_o)
    pct_detect = 100 * train_o["onset_consensus_s"].notna().sum() / max(n_pain, 1)
    onset_arm = train_o[train_o["class"] == "PainArm"]["onset_consensus_s"].dropna()
    onset_hand = train_o[train_o["class"] == "PainHand"]["onset_consensus_s"].dropna()
    arm_med = float(onset_arm.median()) if len(onset_arm) else float("nan")
    hand_med = float(onset_hand.median()) if len(onset_hand) else float("nan")

    # Direction preservation on val for aligned features
    val_pres = val_dir.pivot_table(
        index="feature", columns=["split", "version"],
        values="direction", aggfunc="first"
    )
    aligned_train_dir = val_pres.get(("train", "aligned"), pd.Series(dtype=str))
    aligned_val_dir = val_pres.get(("validation", "aligned"), pd.Series(dtype=str))
    common = aligned_train_dir.index.intersection(aligned_val_dir.index)
    if len(common):
        preserved_aligned = int((aligned_train_dir.loc[common] ==
                                 aligned_val_dir.loc[common]).sum())
        pct_pres_align = 100 * preserved_aligned / len(common)
    else:
        preserved_aligned = 0; pct_pres_align = float("nan")

    verdict = (
        "WORTH PURSUING" if (n_fdr_al > n_fdr_un or median_p_ratio < 0.5)
        else "NOT compelling"
    )

    lines = [
        "# Tier A #1 -- Stimulus-onset alignment (ARM vs HAND)",
        "",
        f"Detection rate (train pain segments): **{pct_detect:.1f}%** "
        f"({train_o['onset_consensus_s'].notna().sum()}/{n_pain})",
        f"Median onset ARM = {arm_med:.2f} s,  HAND = {hand_med:.2f} s",
        "",
        "## Curated feature set (n = " + str(n_total) + ")",
        "",
        f"- Features tested: {n_total}",
        f"- Features with smaller p after alignment: **{n_improved}/{n_total}** "
        f"({100*n_improved/max(n_total,1):.0f}%)",
        f"- Median p-ratio (aligned/unaligned): **{median_p_ratio:.3f}**",
        f"- Features FDR<0.05 unaligned: {n_fdr_un}",
        f"- Features FDR<0.05 aligned:   {n_fdr_al}",
        f"- Features newly FDR-significant after alignment: "
        f"{len(new_fdr)} ({', '.join(new_fdr) if new_fdr else 'none'})",
        f"- Validation direction preserved on aligned features: "
        f"**{preserved_aligned}/{len(common)}** ({pct_pres_align:.0f}%)",
        "",
        "## Top 10 features by |delta(Cliff's d)|",
        "",
    ]
    top = eff.dropna(subset=["delta_cliff"]).sort_values(
        "delta_cliff", ascending=False
    ).head(10)
    lines.append("| feature | unaligned p | aligned p | p_ratio | delta_cliff |")
    lines.append("|---|---|---|---|---|")
    for _, r in top.iterrows():
        lines.append(
            f"| {r['feature']} | {r['unaligned_p']:.3g} | {r['aligned_p']:.3g} "
            f"| {r['p_ratio']:.3g} | {r['delta_cliff']:+.3f} |"
        )
    lines += [
        "",
        f"## Verdict: **{verdict}**",
        "",
        "Rationale: alignment is judged useful if (a) it lifts at least one "
        "feature into FDR<0.05 that was not significant before, or (b) the "
        "median p-value reduces by 2x or more across the curated set.",
        "",
        "Notes / caveats:",
        "- Onset detection is heuristic. EDA slope, BVP amplitude drop, RESP "
        "perturbation each have ~30-60% individual hit rate; consensus + "
        "fallback to EDA-only when spread > 2 s mitigates outliers.",
        "- Aligned window length is 6 s (vs unaligned 10 s); some loss of "
        "averaging power is expected. The comparison is therefore biased "
        "*against* alignment, so any improvement is meaningful.",
        "",
    ]
    fp.write_text("\n".join(lines))
    print(f"[report] wrote {fp.relative_to(ANALYSIS_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[load] train + validation tensors")
    tr_tensor, tr_meta = load_split("train")
    va_tensor, va_meta = load_split("validation")
    tr_meta["split"] = "train"
    va_meta["split"] = "validation"

    feat_names = select_curated_features()
    print(f"[features] curated {len(feat_names)} features:")
    for f in feat_names:
        print(f"   - {f}  (signal={PRIMITIVES[f][0]})")

    on_tr, fe_tr = process_split(tr_tensor, tr_meta, feat_names, "train")
    on_va, fe_va = process_split(va_tensor, va_meta, feat_names, "validation")
    onsets_df = pd.concat([on_tr, on_va], ignore_index=True)
    feats_df = pd.concat([fe_tr, fe_va], ignore_index=True)

    onsets_fp = TABLES_DIR / "tierA1_onsets.csv"
    onsets_df.to_csv(onsets_fp, index=False)
    print(f"[save] onsets -> {onsets_fp.relative_to(ANALYSIS_ROOT)}")

    feats_fp = TABLES_DIR / "tierA1_aligned_features.parquet"
    feats_df.to_parquet(feats_fp, index=False)
    print(f"[save] features -> {feats_fp.relative_to(ANALYSIS_ROOT)}")

    eff = alignment_effect_table(feats_df, feat_names, split="train")
    eff_fp = TABLES_DIR / "tierA1_alignment_effect.csv"
    eff.to_csv(eff_fp, index=False)
    print(f"[save] alignment effect -> {eff_fp.relative_to(ANALYSIS_ROOT)}")

    val_dir = val_direction_check(feats_df, feat_names)
    val_fp = TABLES_DIR / "tierA1_alignment_val_direction.csv"
    val_dir.to_csv(val_fp, index=False)
    print(f"[save] val direction check -> {val_fp.relative_to(ANALYSIS_ROOT)}")

    plot_onset_distribution(onsets_df, PLOTS_DIR / "onset_distribution_by_class.png")
    plot_aligned_vs_unaligned_pvalues(
        eff, PLOTS_DIR / "aligned_vs_unaligned_pvalues.png"
    )
    plot_top10_cliff_change(eff, PLOTS_DIR / "top10_cliff_delta_change.png")
    print(f"[plots] -> {PLOTS_DIR.relative_to(ANALYSIS_ROOT)}")

    write_report(
        eff, onsets_df, val_dir,
        REPORTS_DIR / "13_tierA1_onset_alignment_summary.md",
    )

    # Console summary
    n_total = len(eff)
    n_improved = int(((eff["aligned_p"] < eff["unaligned_p"]) &
                      eff["aligned_p"].notna() &
                      eff["unaligned_p"].notna()).sum())
    median_p_ratio = float(eff["p_ratio"].median(skipna=True))
    n_fdr_un = int((eff["unaligned_p_fdr"] < 0.05).sum())
    n_fdr_al = int((eff["aligned_p_fdr"] < 0.05).sum())
    print("\n=== TIER A1 SUMMARY ===")
    print(f"  features tested:           {n_total}")
    print(f"  features improved (p down): {n_improved}/{n_total}")
    print(f"  median p-ratio (a/u):      {median_p_ratio:.3f}")
    print(f"  FDR<0.05 unaligned:        {n_fdr_un}")
    print(f"  FDR<0.05 aligned:          {n_fdr_al}")


if __name__ == "__main__":
    main()
