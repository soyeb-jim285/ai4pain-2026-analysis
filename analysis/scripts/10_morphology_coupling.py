"""PPG-morphology + cross-signal coupling features for AI4Pain 2026.

Purpose
-------
Previous pass (06_class_tests.py) found 0/243 aggregate features surviving
FDR for PainArm vs PainHand. This script probes a different surface:
  - PPG (BVP) waveform morphology (systolic peak, dicrotic notch, rise/fall,
    augmentation index, 2nd-derivative acceleration peaks).
  - BVP envelope / HR-vs-RESP phase coupling (PLV, Kuiper V, RSA proxy).
  - BVP-envelope vs EDA lagged cross-correlation.
  - RESP-EDA coupling (raw + bandpassed r, lagged xcorr).
  - Mutual information + distance correlation across the 6 signal pairs.

Subject-z normalisation is mandatory (applied per subject, per signal).

Outputs
-------
  results/tables/morphology_coupling_features.parquet
  results/tables/morphology_coupling_tests.csv
  results/reports/10_morphology_coupling_summary.md
  plots/armhand_morphology/*.png

Run:
    uv run python scripts/10_morphology_coupling.py
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
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

try:
    import statsmodels.formula.api as smf  # noqa: F401
    from statsmodels.regression.mixed_linear_model import MixedLM  # noqa: F401
    HAVE_LMM = True
except Exception:
    HAVE_LMM = False

try:
    import dcor  # type: ignore

    HAVE_DCOR = True
except Exception:
    HAVE_DCOR = False

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_DIR / "results" / "tables"
REPORTS_DIR = ANALYSIS_DIR / "results" / "reports"
PLOTS_DIR = ANALYSIS_DIR / "plots" / "armhand_morphology"
for d in (TABLES_DIR, REPORTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
BVP_IDX = SIGNALS.index("Bvp")
EDA_IDX = SIGNALS.index("Eda")
RESP_IDX = SIGNALS.index("Resp")
SPO2_IDX = SIGNALS.index("SpO2")


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _strip_nan(x: np.ndarray) -> np.ndarray:
    """Strip trailing NaN, linearly interpolate any interior NaN."""
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
        x = np.interp(idx, idx[good], x[good]).astype(np.float32)
    return x.astype(np.float32)


def _butter_filt(
    x: np.ndarray,
    low: float | None,
    high: float | None,
    sfreq: float = SFREQ,
    order: int = 3,
) -> np.ndarray:
    if len(x) < max(3 * order + 1, 15):
        return x.astype(np.float32)
    nyq = sfreq / 2.0
    if low is not None and high is not None:
        b, a = spsig.butter(order, [low / nyq, high / nyq], btype="band")
    elif low is not None:
        b, a = spsig.butter(order, low / nyq, btype="high")
    elif high is not None:
        b, a = spsig.butter(order, high / nyq, btype="low")
    else:
        return x.astype(np.float32)
    try:
        y = spsig.filtfilt(b, a, x, padlen=min(3 * max(len(a), len(b)), len(x) - 1))
    except Exception:
        return x.astype(np.float32)
    return y.astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    if s <= 0 or not np.isfinite(s):
        return x - np.mean(x)
    return (x - np.mean(x)) / s


# ---------------------------------------------------------------------------
# Subject-z normalisation
# ---------------------------------------------------------------------------
def subject_z_normalise(
    tensor: np.ndarray, meta: pd.DataFrame
) -> np.ndarray:
    """Per subject, per signal: subtract the subject+signal mean and divide by
    its std, computed across all (segment, sample) pairs for that subject."""
    out = tensor.copy()
    for sid, idxs in meta.groupby("subject").indices.items():
        sub = out[list(idxs)]  # (n_seg_subj, 4, T)
        for s in range(sub.shape[1]):
            flat = sub[:, s, :].reshape(-1)
            flat = flat[~np.isnan(flat)]
            if len(flat) < 10:
                continue
            mu = float(np.mean(flat))
            sd = float(np.std(flat))
            if sd <= 0 or not np.isfinite(sd):
                sd = 1.0
            out[list(idxs), s, :] = (out[list(idxs), s, :] - mu) / sd
    return out


# ---------------------------------------------------------------------------
# PPG morphology
# ---------------------------------------------------------------------------
def ppg_morphology(bvp: np.ndarray, sfreq: float = SFREQ) -> dict[str, float]:
    """Per-segment PPG waveform morphology features."""
    feats: dict[str, float] = {
        "ppg_n_beats": np.nan,
        "ppg_sys_amp_mean": np.nan,
        "ppg_sys_amp_std": np.nan,
        "ppg_trough_mean": np.nan,
        "ppg_trough_std": np.nan,
        "ppg_pk2tr_amp_mean": np.nan,
        "ppg_pk2tr_amp_std": np.nan,
        "ppg_halfwidth_mean": np.nan,
        "ppg_halfwidth_std": np.nan,
        "ppg_rise_time_mean": np.nan,
        "ppg_rise_time_std": np.nan,
        "ppg_fall_time_mean": np.nan,
        "ppg_fall_time_std": np.nan,
        "ppg_prominence_mean": np.nan,
        "ppg_prominence_std": np.nan,
        "ppg_notch_depth_rel_mean": np.nan,
        "ppg_notch_depth_rel_std": np.nan,
        "ppg_notch_time_frac_mean": np.nan,
        "ppg_notch_time_frac_std": np.nan,
        "ppg_refl_index_mean": np.nan,
        "ppg_refl_index_std": np.nan,
        "ppg_rise_frac_mean": np.nan,
        "ppg_rise_frac_std": np.nan,
        "ppg_aug_index_mean": np.nan,
        "ppg_aug_index_std": np.nan,
        "ppg_d2_peak_count": np.nan,
        "ppg_d2_peak_count_per_beat": np.nan,
    }
    if len(bvp) < int(1.5 * sfreq):
        return feats
    try:
        x = _butter_filt(bvp, 0.5, 8.0, sfreq)
        min_dist = int(0.4 * sfreq)  # 150 bpm ceiling
        # Use prominence relative to signal std for robustness.
        prom = 0.3 * float(np.std(x)) if np.std(x) > 0 else None
        peaks, pk_props = spsig.find_peaks(x, distance=min_dist, prominence=prom)
        if len(peaks) < 4:
            feats["ppg_n_beats"] = float(len(peaks))
            return feats
        # Troughs: minima of -x between successive peaks
        troughs: list[int] = []
        for i in range(len(peaks) - 1):
            seg = x[peaks[i] : peaks[i + 1]]
            if len(seg) < 3:
                troughs.append(int(peaks[i]))
                continue
            troughs.append(int(peaks[i] + int(np.argmin(seg))))
        # Also one prior to first peak (if room)
        pre_start = max(0, peaks[0] - min_dist * 2)
        if peaks[0] - pre_start > 3:
            tr0 = pre_start + int(np.argmin(x[pre_start : peaks[0]]))
        else:
            tr0 = pre_start
        troughs = [tr0] + troughs

        # Per-beat features: use peak i (which has trough_prev = troughs[i] and
        # trough_next = troughs[i+1] if available).
        dx = np.diff(x)
        sys_amps: list[float] = []
        tr_vals: list[float] = []
        pk2tr: list[float] = []
        halfwidths: list[float] = []
        rise_t: list[float] = []
        fall_t: list[float] = []
        proms: list[float] = []
        notch_depth_rel: list[float] = []
        notch_time_frac: list[float] = []
        refl_index: list[float] = []
        rise_frac: list[float] = []
        aug_index: list[float] = []

        prom_arr = pk_props.get("prominences", np.full(len(peaks), np.nan))

        for i, pk in enumerate(peaks):
            if i + 1 >= len(troughs):
                break
            tr_prev = troughs[i]
            tr_next = troughs[i + 1]
            if tr_next <= pk or pk <= tr_prev:
                continue
            amp = float(x[pk])
            tr_val = float(x[tr_next])
            pk_tr = amp - tr_val
            if pk_tr <= 0 or not np.isfinite(pk_tr):
                continue
            sys_amps.append(amp)
            tr_vals.append(tr_val)
            pk2tr.append(pk_tr)
            proms.append(float(prom_arr[i]) if i < len(prom_arr) else np.nan)

            beat_dur = (tr_next - tr_prev) / sfreq
            rise = (pk - tr_prev) / sfreq
            fall = (tr_next - pk) / sfreq
            rise_t.append(rise)
            fall_t.append(fall)
            rise_frac.append(rise / beat_dur if beat_dur > 0 else np.nan)

            # Half-height width within this beat
            half_level = tr_val + 0.5 * pk_tr
            left = pk
            while left > tr_prev and x[left] > half_level:
                left -= 1
            right = pk
            while right < tr_next and x[right] > half_level:
                right += 1
            halfwidths.append((right - left) / sfreq)

            # Dicrotic notch: local minimum in first-derivative sign change
            # region [pk + 30% beat, pk + 60% beat] relative to inter-beat.
            ibi = tr_next - tr_prev
            notch_lo = pk + int(0.30 * ibi)
            notch_hi = pk + int(0.60 * ibi)
            notch_lo = min(max(notch_lo, pk + 2), tr_next - 2)
            notch_hi = min(max(notch_hi, notch_lo + 2), tr_next - 1)
            notch_found = False
            if notch_hi > notch_lo + 1:
                seg = x[notch_lo : notch_hi + 1]
                # Prefer a local minimum inside the window
                loc_mins, _ = spsig.find_peaks(-seg)
                if len(loc_mins) > 0:
                    notch_idx = notch_lo + int(loc_mins[0])
                else:
                    notch_idx = notch_lo + int(np.argmin(seg))
                notch_val = float(x[notch_idx])
                depth_rel = (amp - notch_val) / pk_tr
                notch_depth_rel.append(depth_rel)
                time_frac = (notch_idx - tr_prev) / ibi if ibi > 0 else np.nan
                notch_time_frac.append(time_frac)
                # Augmentation index proxy: (systolic - notch) / systolic  (on
                # the peak-relative amplitude so it is scale-invariant).
                aug_index.append(depth_rel)
                # Reflection-index proxy: |dx at notch| / |dx max on upstroke|
                up_start = tr_prev
                up_end = pk
                if up_end - up_start > 2 and notch_idx > 0:
                    max_up = float(np.max(np.abs(dx[up_start:up_end])))
                    notch_dx = float(np.abs(dx[min(notch_idx, len(dx) - 1)]))
                    if max_up > 0:
                        refl_index.append(notch_dx / max_up)
                notch_found = True
            if not notch_found:
                notch_depth_rel.append(np.nan)
                notch_time_frac.append(np.nan)
                aug_index.append(np.nan)

        def _mean(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if len(arr) else np.nan

        def _std(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan

        feats["ppg_n_beats"] = float(len(peaks))
        feats["ppg_sys_amp_mean"] = _mean(sys_amps)
        feats["ppg_sys_amp_std"] = _std(sys_amps)
        feats["ppg_trough_mean"] = _mean(tr_vals)
        feats["ppg_trough_std"] = _std(tr_vals)
        feats["ppg_pk2tr_amp_mean"] = _mean(pk2tr)
        feats["ppg_pk2tr_amp_std"] = _std(pk2tr)
        feats["ppg_halfwidth_mean"] = _mean(halfwidths)
        feats["ppg_halfwidth_std"] = _std(halfwidths)
        feats["ppg_rise_time_mean"] = _mean(rise_t)
        feats["ppg_rise_time_std"] = _std(rise_t)
        feats["ppg_fall_time_mean"] = _mean(fall_t)
        feats["ppg_fall_time_std"] = _std(fall_t)
        feats["ppg_prominence_mean"] = _mean(proms)
        feats["ppg_prominence_std"] = _std(proms)
        feats["ppg_notch_depth_rel_mean"] = _mean(notch_depth_rel)
        feats["ppg_notch_depth_rel_std"] = _std(notch_depth_rel)
        feats["ppg_notch_time_frac_mean"] = _mean(notch_time_frac)
        feats["ppg_notch_time_frac_std"] = _std(notch_time_frac)
        feats["ppg_refl_index_mean"] = _mean(refl_index)
        feats["ppg_refl_index_std"] = _std(refl_index)
        feats["ppg_rise_frac_mean"] = _mean(rise_frac)
        feats["ppg_rise_frac_std"] = _std(rise_frac)
        feats["ppg_aug_index_mean"] = _mean(aug_index)
        feats["ppg_aug_index_std"] = _std(aug_index)

        # Second-derivative acceleration peaks of BVP
        d2 = np.diff(x, n=2)
        d2_prom = 0.3 * float(np.std(d2)) if np.std(d2) > 0 else None
        d2_peaks, _ = spsig.find_peaks(d2, distance=int(0.15 * sfreq), prominence=d2_prom)
        feats["ppg_d2_peak_count"] = float(len(d2_peaks))
        feats["ppg_d2_peak_count_per_beat"] = (
            float(len(d2_peaks)) / len(peaks) if len(peaks) else np.nan
        )
    except Exception:
        pass
    return feats


# ---------------------------------------------------------------------------
# Coupling helpers
# ---------------------------------------------------------------------------
def _analytic(x: np.ndarray) -> np.ndarray:
    try:
        return spsig.hilbert(x - np.mean(x))
    except Exception:
        return np.zeros_like(x, dtype=complex)


def _phase(x: np.ndarray) -> np.ndarray:
    return np.angle(_analytic(x))


def _envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(_analytic(x))


def _kuiper_v(phi: np.ndarray) -> float:
    """Kuiper's V statistic on circular data in [-pi, pi]."""
    u = (phi + np.pi) / (2 * np.pi)
    u = np.sort(u)
    n = len(u)
    if n < 2:
        return np.nan
    i = np.arange(1, n + 1)
    D_plus = np.max(i / n - u)
    D_minus = np.max(u - (i - 1) / n)
    return float((D_plus + D_minus) * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)))


def coupling_features(
    bvp: np.ndarray,
    eda: np.ndarray,
    resp: np.ndarray,
    spo2: np.ndarray | None,
    spo2_valid: bool,
    sfreq: float = SFREQ,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    # Ensure aligned lengths
    L = min(len(bvp), len(eda), len(resp))
    if spo2 is not None:
        L = min(L, len(spo2))
    bvp = bvp[:L]
    eda = eda[:L]
    resp = resp[:L]
    if spo2 is not None:
        spo2 = spo2[:L]
    if L < int(3 * sfreq):
        return feats

    # Bandpass
    try:
        bvp_bp = _butter_filt(bvp, 0.5, 4.0, sfreq)
    except Exception:
        bvp_bp = bvp.copy()
    try:
        resp_bp = _butter_filt(resp, 0.1, 0.5, sfreq)
    except Exception:
        resp_bp = resp.copy()
    try:
        eda_lp = _butter_filt(eda, None, 2.0, sfreq)
    except Exception:
        eda_lp = eda.copy()
    try:
        resp_eda_bp = _butter_filt(resp, 0.05, 0.5, sfreq)
    except Exception:
        resp_eda_bp = resp.copy()
    try:
        eda_slow_bp = _butter_filt(eda, 0.05, 0.5, sfreq)
    except Exception:
        eda_slow_bp = eda.copy()

    # ---------- RESP-BVP phase coupling (envelope of BVP vs RESP) ----------
    try:
        bvp_env = _envelope(bvp_bp)
        # Bandpass envelope to respiratory rhythm
        bvp_env_bp = _butter_filt(bvp_env - np.mean(bvp_env), 0.1, 0.5, sfreq)
        phi_bvp_slow = _phase(bvp_env_bp)
        phi_resp = _phase(resp_bp)
        dphi = np.mod(phi_bvp_slow - phi_resp + np.pi, 2 * np.pi) - np.pi
        plv = float(np.abs(np.mean(np.exp(1j * (phi_bvp_slow - phi_resp)))))
        mean_phase = float(spstats.circmean(dphi, high=np.pi, low=-np.pi))
        # Decimate dphi to reduce autocorrelation for Kuiper V
        step = max(1, int(sfreq // 5))
        feats["resp_bvp_plv"] = plv
        feats["resp_bvp_mean_phase_diff"] = mean_phase
        feats["resp_bvp_kuiper_V"] = _kuiper_v(dphi[::step])
    except Exception:
        feats["resp_bvp_plv"] = np.nan
        feats["resp_bvp_mean_phase_diff"] = np.nan
        feats["resp_bvp_kuiper_V"] = np.nan

    # ---------- HR-RSA proxy ----------
    try:
        min_dist = int(0.4 * sfreq)
        prom = 0.3 * float(np.std(bvp_bp)) if np.std(bvp_bp) > 0 else None
        peaks, _ = spsig.find_peaks(bvp_bp, distance=min_dist, prominence=prom)
        if len(peaks) >= 4:
            hr_times = peaks[1:] / sfreq
            ibi = np.diff(peaks) / sfreq
            hr_inst = 60.0 / ibi
            # Interpolate onto regular grid at sfreq
            t_reg = np.arange(L) / sfreq
            # Only within the hr_times support
            lo, hi = hr_times[0], hr_times[-1]
            mask = (t_reg >= lo) & (t_reg <= hi)
            if mask.sum() > 20:
                hr_reg = np.interp(t_reg[mask], hr_times, hr_inst)
                resp_seg = resp_bp[mask]
                if np.std(hr_reg) > 0 and np.std(resp_seg) > 0:
                    r, _ = spstats.pearsonr(hr_reg, resp_seg)
                    feats["rsa_hr_resp_r"] = float(r)
                else:
                    feats["rsa_hr_resp_r"] = np.nan
            else:
                feats["rsa_hr_resp_r"] = np.nan
        else:
            feats["rsa_hr_resp_r"] = np.nan
    except Exception:
        feats["rsa_hr_resp_r"] = np.nan

    # ---------- BVP envelope vs EDA lagged cross-correlation ----------
    try:
        env = _envelope(bvp_bp)
        env_z = _zscore(env)
        eda_z = _zscore(eda_lp)
        max_lag = int(5.0 * sfreq)
        step = max(1, int(0.05 * sfreq))
        lags = np.arange(-max_lag, max_lag + 1, step)
        corrs = np.zeros(len(lags), dtype=float)
        for i, lag in enumerate(lags):
            if lag >= 0:
                a = env_z[lag:]
                b = eda_z[: len(a)]
            else:
                b = eda_z[-lag:]
                a = env_z[: len(b)]
            if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
                corrs[i] = float(np.corrcoef(a, b)[0, 1])
            else:
                corrs[i] = np.nan
        if np.isfinite(corrs).any():
            idx = int(np.nanargmax(np.abs(corrs)))
            feats["bvp_env_eda_xcorr_max"] = float(corrs[idx])
            feats["bvp_env_eda_xcorr_lag"] = float(lags[idx] / sfreq)
            feats["bvp_env_eda_xcorr_at_zero"] = float(
                corrs[int(np.argmin(np.abs(lags)))]
            )
        else:
            feats["bvp_env_eda_xcorr_max"] = np.nan
            feats["bvp_env_eda_xcorr_lag"] = np.nan
            feats["bvp_env_eda_xcorr_at_zero"] = np.nan
    except Exception:
        feats["bvp_env_eda_xcorr_max"] = np.nan
        feats["bvp_env_eda_xcorr_lag"] = np.nan
        feats["bvp_env_eda_xcorr_at_zero"] = np.nan

    # ---------- BVP envelope slow-phase vs EDA slow-phase correlation ----------
    try:
        env_slow = _butter_filt(_envelope(bvp_bp), 0.05, 0.5, sfreq)
        phi_env = _phase(env_slow)
        phi_eda = _phase(eda_slow_bp)
        dphi = np.mod(phi_env - phi_eda + np.pi, 2 * np.pi) - np.pi
        feats["bvp_env_eda_phase_plv"] = float(
            np.abs(np.mean(np.exp(1j * (phi_env - phi_eda))))
        )
        feats["bvp_env_eda_phase_mean"] = float(
            spstats.circmean(dphi, high=np.pi, low=-np.pi)
        )
    except Exception:
        feats["bvp_env_eda_phase_plv"] = np.nan
        feats["bvp_env_eda_phase_mean"] = np.nan

    # ---------- RESP-EDA coupling ----------
    try:
        if np.std(resp) > 0 and np.std(eda) > 0:
            r_raw, _ = spstats.pearsonr(resp, eda)
            feats["resp_eda_r_raw"] = float(r_raw)
        else:
            feats["resp_eda_r_raw"] = np.nan
        if np.std(resp_eda_bp) > 0 and np.std(eda_slow_bp) > 0:
            r_bp, _ = spstats.pearsonr(resp_eda_bp, eda_slow_bp)
            feats["resp_eda_r_bp"] = float(r_bp)
        else:
            feats["resp_eda_r_bp"] = np.nan
    except Exception:
        feats["resp_eda_r_raw"] = np.nan
        feats["resp_eda_r_bp"] = np.nan

    try:
        max_lag = int(5.0 * sfreq)
        step = max(1, int(0.05 * sfreq))
        lags = np.arange(-max_lag, max_lag + 1, step)
        a0 = _zscore(resp_eda_bp)
        b0 = _zscore(eda_slow_bp)
        corrs = np.zeros(len(lags), dtype=float)
        for i, lag in enumerate(lags):
            if lag >= 0:
                a = a0[lag:]
                b = b0[: len(a)]
            else:
                b = b0[-lag:]
                a = a0[: len(b)]
            if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
                corrs[i] = float(np.corrcoef(a, b)[0, 1])
            else:
                corrs[i] = np.nan
        if np.isfinite(corrs).any():
            idx = int(np.nanargmax(np.abs(corrs)))
            feats["resp_eda_xcorr_max"] = float(corrs[idx])
            feats["resp_eda_xcorr_lag"] = float(lags[idx] / sfreq)
        else:
            feats["resp_eda_xcorr_max"] = np.nan
            feats["resp_eda_xcorr_lag"] = np.nan
    except Exception:
        feats["resp_eda_xcorr_max"] = np.nan
        feats["resp_eda_xcorr_lag"] = np.nan

    # ---------- Mutual information + distance correlation across pairs ----------
    sig_map: dict[str, np.ndarray] = {
        "bvp": bvp,
        "eda": eda,
        "resp": resp,
    }
    if spo2 is not None and spo2_valid:
        sig_map["spo2"] = spo2
    pair_names = list(combinations(sig_map.keys(), 2))
    for a, b in pair_names:
        x = sig_map[a]
        y = sig_map[b]
        try:
            mi = _mutual_information(x, y, bins=32)
            feats[f"mi_{a}_{b}"] = mi
        except Exception:
            feats[f"mi_{a}_{b}"] = np.nan
        try:
            dc = _distance_correlation(x, y, subsample=5)
            feats[f"dcor_{a}_{b}"] = dc
        except Exception:
            feats[f"dcor_{a}_{b}"] = np.nan
    # Fill missing pairs with NaN if SpO2 invalid so the feature matrix is
    # rectangular across segments.
    for a, b in combinations(["bvp", "eda", "resp", "spo2"], 2):
        feats.setdefault(f"mi_{a}_{b}", np.nan)
        feats.setdefault(f"dcor_{a}_{b}", np.nan)

    return feats


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 20 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = hist / hist.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mi = np.nansum(
            p_xy * (np.log(p_xy + 1e-12) - np.log(p_x + 1e-12) - np.log(p_y + 1e-12))
        )
    return float(max(mi, 0.0))


def _distance_correlation(x: np.ndarray, y: np.ndarray, subsample: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 20 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if HAVE_DCOR:
        try:
            return float(dcor.distance_correlation(x[::subsample], y[::subsample]))
        except Exception:
            pass
    # Manual O(n^2) on downsampled data
    xs = x[::subsample]
    ys = y[::subsample]
    n = len(xs)
    if n < 10:
        return np.nan
    a = np.abs(xs[:, None] - xs[None, :])
    b = np.abs(ys[:, None] - ys[None, :])
    a_mean_r = a.mean(axis=1, keepdims=True)
    a_mean_c = a.mean(axis=0, keepdims=True)
    a_grand = a.mean()
    b_mean_r = b.mean(axis=1, keepdims=True)
    b_mean_c = b.mean(axis=0, keepdims=True)
    b_grand = b.mean()
    A = a - a_mean_r - a_mean_c + a_grand
    B = b - b_mean_r - b_mean_c + b_grand
    dcov2_xy = (A * B).mean()
    dcov2_xx = (A * A).mean()
    dcov2_yy = (B * B).mean()
    if dcov2_xx <= 0 or dcov2_yy <= 0 or dcov2_xy < 0:
        return 0.0
    return float(np.sqrt(dcov2_xy / np.sqrt(dcov2_xx * dcov2_yy)))


# ---------------------------------------------------------------------------
# Per-segment feature extraction
# ---------------------------------------------------------------------------
def extract_segment_features(
    seg: np.ndarray, spo2_valid: bool
) -> dict[str, float]:
    bvp = _strip_nan(seg[BVP_IDX])
    eda = _strip_nan(seg[EDA_IDX])
    resp = _strip_nan(seg[RESP_IDX])
    spo2 = _strip_nan(seg[SPO2_IDX])
    feats: dict[str, float] = {}
    try:
        feats.update(ppg_morphology(bvp))
    except Exception:
        pass
    try:
        feats.update(coupling_features(bvp, eda, resp, spo2, spo2_valid))
    except Exception:
        pass
    return feats


# ---------------------------------------------------------------------------
# Build feature matrix for both splits
# ---------------------------------------------------------------------------
def build_feature_matrix() -> pd.DataFrame:
    # Load constant-SpO2 inventory once
    const_df = pd.read_csv(TABLES_DIR / "inventory_constant_segments.csv")
    const_set = set(
        const_df[const_df["signal"] == "SpO2"]["segment_id"].astype(str).tolist()
    )

    all_rows: list[dict] = []
    for split in ("train", "validation"):
        print(f"[load] split={split}")
        tensor, meta = load_split(split)
        print(f"  tensor {tensor.shape}, meta {meta.shape}")
        # Subject-z normalise
        print("  subject-z normalising...")
        tensor = subject_z_normalise(tensor, meta)
        n = len(meta)
        for i in tqdm(range(n), desc=f"feat {split}", ncols=88):
            row = meta.iloc[i]
            seg_id = str(row["segment_id"])
            spo2_valid = seg_id not in const_set
            feats = extract_segment_features(tensor[i], spo2_valid)
            out = {k: row[k] for k in META_COLS}
            out.update(feats)
            all_rows.append(out)
    df = pd.DataFrame(all_rows)
    return df


# ---------------------------------------------------------------------------
# ARM vs HAND statistical tests
# ---------------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    gt = int(np.sum(x[:, None] > y[None, :]))
    lt = int(np.sum(x[:, None] < y[None, :]))
    return float(gt - lt) / (nx * ny)


def paired_subject_means(
    df: pd.DataFrame, feature: str
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
            f"{feature} ~ is_hand", groups="subject", data=sub
        )
        mdf = md.fit(method="lbfgs", reml=True, disp=False)
        p = float(mdf.pvalues.get("is_hand", np.nan))
        return p
    except Exception:
        return np.nan


def run_armhand_tests(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    tr = df[df["split"] == "train"].copy()
    rows: list[dict] = []
    for feat in tqdm(features, desc="armhand tests", ncols=88):
        x, y, _ = paired_subject_means(tr, feat)
        row = {"feature": feat, "n_subjects": int(len(x))}
        # Paired Wilcoxon
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
        # Cliff's delta (unpaired flavor)
        row["cliffs_delta"] = cliffs_delta(x, y)
        # Sign consistency
        d = x - y
        d = d[np.isfinite(d) & (d != 0)]
        if len(d) > 0:
            pos = int((d > 0).sum())
            neg = int((d < 0).sum())
            row["n_pos"] = pos
            row["n_neg"] = neg
            row["sign_consistency"] = max(pos, neg) / len(d)
        else:
            row["n_pos"] = 0
            row["n_neg"] = 0
            row["sign_consistency"] = np.nan
        row["mean_arm"] = float(np.mean(x)) if len(x) else np.nan
        row["mean_hand"] = float(np.mean(y)) if len(y) else np.nan
        row["direction"] = (
            "Arm > Hand" if row["mean_arm"] > row["mean_hand"] else "Arm < Hand"
        )
        # LMM
        row["lmm_p"] = fit_lmm(tr, feat)
        rows.append(row)
    out = pd.DataFrame(rows)

    # Feature families: morphology vs coupling
    def _fam(f: str) -> str:
        if f.startswith("ppg_"):
            return "morphology"
        if f.startswith("mi_") or f.startswith("dcor_"):
            return "coupling_nonlinear"
        return "coupling_linear"

    out["family"] = out["feature"].apply(_fam)

    # FDR per family per test (wilcoxon, lmm)
    for test_col in ["wilcoxon_p", "lmm_p"]:
        adj_col = test_col.replace("_p", "_p_fdr")
        out[adj_col] = np.nan
        for fam, sub in out.groupby("family"):
            p = sub[test_col].to_numpy()
            mask = np.isfinite(p)
            if mask.sum() == 0:
                continue
            _, p_adj, _, _ = multipletests(p[mask], alpha=0.05, method="fdr_bh")
            adj = np.full_like(p, np.nan, dtype=float)
            adj[mask] = p_adj
            out.loc[sub.index, adj_col] = adj

    # Global FDR across all features for Wilcoxon (comparison with family-wise)
    p = out["wilcoxon_p"].to_numpy()
    mask = np.isfinite(p)
    adj_all = np.full_like(p, np.nan, dtype=float)
    if mask.sum() > 0:
        _, p_adj, _, _ = multipletests(p[mask], alpha=0.05, method="fdr_bh")
        adj_all[mask] = p_adj
    out["wilcoxon_p_fdr_global"] = adj_all

    return out


def validation_reproducibility(
    df: pd.DataFrame, top_feats: list[str]
) -> pd.DataFrame:
    val = df[df["split"] == "validation"]
    tr = df[df["split"] == "train"]
    rows: list[dict] = []
    for feat in top_feats:
        tr_arm, tr_hand, _ = paired_subject_means(tr, feat)
        va_arm, va_hand, _ = paired_subject_means(val, feat)
        tr_eff = float(np.mean(tr_arm) - np.mean(tr_hand)) if len(tr_arm) else np.nan
        va_eff = float(np.mean(va_arm) - np.mean(va_hand)) if len(va_arm) else np.nan
        preserved = (np.sign(tr_eff) == np.sign(va_eff)) and np.isfinite(tr_eff) and np.isfinite(va_eff)
        rows.append(
            {
                "feature": feat,
                "train_arm_minus_hand": tr_eff,
                "val_arm_minus_hand": va_eff,
                "train_direction_preserved_in_val": bool(preserved),
                "val_n_subjects": int(len(va_arm)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_ppg_beat_template(df_feat: pd.DataFrame, tensor_tr: np.ndarray, meta_tr: pd.DataFrame) -> None:
    """Per-class average beat template aligned on systolic peak."""
    fig, ax = plt.subplots(figsize=(8, 5))
    subjects = sorted(meta_tr["subject"].unique())
    win = int(0.5 * SFREQ)  # +/- 500 ms

    for cls, color in [("PainArm", "#dd8452"), ("PainHand", "#c44e52")]:
        subj_templates: list[np.ndarray] = []
        for sid in subjects:
            mask = (meta_tr["subject"] == sid) & (meta_tr["class"] == cls)
            idxs = meta_tr.index[mask].to_numpy()
            beats: list[np.ndarray] = []
            for i in idxs:
                bvp = _strip_nan(tensor_tr[i, BVP_IDX])
                if len(bvp) < 200:
                    continue
                x = _butter_filt(bvp, 0.5, 8.0, SFREQ)
                x = _zscore(x)
                prom = 0.3 * float(np.std(x)) if np.std(x) > 0 else None
                peaks, _ = spsig.find_peaks(x, distance=int(0.4 * SFREQ), prominence=prom)
                for pk in peaks:
                    lo = pk - win
                    hi = pk + win
                    if lo < 0 or hi > len(x):
                        continue
                    beats.append(x[lo:hi])
            if beats:
                subj_templates.append(np.mean(np.stack(beats, axis=0), axis=0))
        if not subj_templates:
            continue
        arr = np.stack(subj_templates, axis=0)
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        t = np.arange(-win, win) / SFREQ
        ax.plot(t, mean, color=color, linewidth=2, label=f"{cls} (n={arr.shape[0]} subj)")
        ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.25)
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time relative to systolic peak (s)")
    ax.set_ylabel("Subject-z BVP")
    ax.set_title("Mean PPG beat template: PainArm vs PainHand (train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ppg_beat_template_per_class.png")
    plt.close(fig)


def plot_resp_bvp_phase_hist(df_feat: pd.DataFrame) -> None:
    tr = df_feat[df_feat["split"] == "train"]
    arm = tr[tr["class"] == "PainArm"]["resp_bvp_mean_phase_diff"].dropna().to_numpy()
    hand = tr[tr["class"] == "PainHand"]["resp_bvp_mean_phase_diff"].dropna().to_numpy()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5), subplot_kw={"projection": "polar"}
    )
    for ax, vals, cls, color in [
        (ax1, arm, "PainArm", "#dd8452"),
        (ax2, hand, "PainHand", "#c44e52"),
    ]:
        bins = np.linspace(-np.pi, np.pi, 25)
        counts, edges = np.histogram(vals, bins=bins)
        widths = np.diff(edges)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts, width=widths, color=color, alpha=0.7, edgecolor="k")
        ax.set_title(f"{cls} (n={len(vals)} segments)")
    fig.suptitle("RESP-BVP mean phase difference distribution")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "resp_bvp_phase_hist.png")
    plt.close(fig)


def plot_lagged_xcorr_bvp_eda(
    tensor_tr: np.ndarray, meta_tr: pd.DataFrame
) -> None:
    subjects = sorted(meta_tr["subject"].unique())
    max_lag = int(5.0 * SFREQ)
    step = max(1, int(0.05 * SFREQ))
    lags = np.arange(-max_lag, max_lag + 1, step)
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls, color in [("PainArm", "#dd8452"), ("PainHand", "#c44e52")]:
        subj_curves: list[np.ndarray] = []
        for sid in subjects:
            mask = (meta_tr["subject"] == sid) & (meta_tr["class"] == cls)
            idxs = meta_tr.index[mask].to_numpy()
            seg_curves: list[np.ndarray] = []
            for i in idxs:
                bvp = _strip_nan(tensor_tr[i, BVP_IDX])
                eda = _strip_nan(tensor_tr[i, EDA_IDX])
                L = min(len(bvp), len(eda))
                if L < int(3 * SFREQ):
                    continue
                bvp = bvp[:L]
                eda = eda[:L]
                try:
                    env = _envelope(_butter_filt(bvp, 0.5, 4.0, SFREQ))
                    eda_lp = _butter_filt(eda, None, 2.0, SFREQ)
                    a0 = _zscore(env)
                    b0 = _zscore(eda_lp)
                    corrs = np.zeros(len(lags), dtype=float)
                    for k, lag in enumerate(lags):
                        if lag >= 0:
                            a = a0[lag:]
                            b = b0[: len(a)]
                        else:
                            b = b0[-lag:]
                            a = a0[: len(b)]
                        if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
                            corrs[k] = float(np.corrcoef(a, b)[0, 1])
                        else:
                            corrs[k] = np.nan
                    seg_curves.append(corrs)
                except Exception:
                    continue
            if seg_curves:
                subj_curves.append(
                    np.nanmean(np.stack(seg_curves, axis=0), axis=0)
                )
        if not subj_curves:
            continue
        arr = np.stack(subj_curves, axis=0)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        t = lags / SFREQ
        ax.plot(t, mean, color=color, linewidth=2, label=f"{cls} (n={arr.shape[0]} subj)")
        ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.25)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lag (s) — positive = BVP-env leads EDA")
    ax.set_ylabel("Pearson r")
    ax.set_title("BVP envelope vs EDA cross-correlation (train, subject-averaged)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lagged_xcorr_bvp_eda.png")
    plt.close(fig)


def plot_sync_index_violin(df_feat: pd.DataFrame) -> None:
    tr = df_feat[df_feat["split"] == "train"]
    sub = tr[tr["class"].isin(["PainArm", "PainHand"])]
    if "resp_bvp_plv" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(
        data=sub, x="class", y="resp_bvp_plv",
        palette={"PainArm": "#dd8452", "PainHand": "#c44e52"}, ax=ax, inner="quartile"
    )
    ax.set_title("RESP-BVP phase-locking value (PLV)")
    ax.set_ylabel("PLV")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "synchronization_index_violin.png")
    plt.close(fig)


def plot_mi_heatmap(df_feat: pd.DataFrame) -> None:
    tr = df_feat[df_feat["split"] == "train"]
    signals4 = ["bvp", "eda", "resp", "spo2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cls in zip(axes, ["PainArm", "PainHand"]):
        mat = np.full((4, 4), np.nan)
        sub = tr[tr["class"] == cls]
        for i, a in enumerate(signals4):
            for j, b in enumerate(signals4):
                if i == j:
                    mat[i, j] = np.nan
                    continue
                key = f"mi_{a}_{b}" if f"mi_{a}_{b}" in tr.columns else f"mi_{b}_{a}"
                if key in sub.columns:
                    mat[i, j] = sub[key].mean()
        sns.heatmap(
            mat, annot=True, fmt=".3f", xticklabels=signals4, yticklabels=signals4,
            cmap="viridis", ax=ax, cbar_kws={"label": "MI (nats)"}
        )
        ax.set_title(f"{cls} — class-mean MI")
    fig.suptitle("Mutual information between signal pairs (train subject-means)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mi_heatmap.png")
    plt.close(fig)


def plot_top20_wilcoxon_bar(tests: pd.DataFrame) -> None:
    t = tests.dropna(subset=["wilcoxon_W"]).copy()
    t["abs_W"] = t["wilcoxon_W"].abs()
    t = t.sort_values("abs_W", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [
        {"morphology": "#4c72b0", "coupling_linear": "#55a868", "coupling_nonlinear": "#c44e52"}.get(
            f, "#888"
        )
        for f in t["family"]
    ]
    ax.barh(t["feature"][::-1], t["abs_W"][::-1], color=colors[::-1])
    ax.set_xlabel("|Paired Wilcoxon W|")
    ax.set_title("Top 20 features by |W| (ARM vs HAND, train)")
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#4c72b0", label="morphology"),
        Patch(color="#55a868", label="coupling (linear)"),
        Patch(color="#c44e52", label="coupling (nonlinear)"),
    ]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top20_morphology_anova_bar.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(
    tests: pd.DataFrame, val_df: pd.DataFrame, n_features: int, n_train_subj: int
) -> None:
    fdr_sig = tests[tests["wilcoxon_p_fdr"] < 0.05]
    fdr_sig_global = tests[tests["wilcoxon_p_fdr_global"] < 0.05]

    top15 = tests.sort_values("wilcoxon_p").head(15).copy()

    # Family summary
    fam_sum = tests.groupby("family").agg(
        n=("feature", "count"),
        n_sig_fdr=("wilcoxon_p_fdr", lambda s: int((s < 0.05).sum())),
        min_p=("wilcoxon_p", "min"),
        median_abs_delta=("cliffs_delta", lambda s: float(np.nanmedian(np.abs(s)))),
    ).reset_index()

    val_lookup = val_df.set_index("feature") if len(val_df) else None

    lines: list[str] = []
    lines.append("# 10 Morphology + coupling features (PainArm vs PainHand)")
    lines.append("")
    lines.append(
        f"- Total features: **{n_features}** (morphology + linear coupling + nonlinear coupling)"
    )
    lines.append(
        f"- Train subjects (paired Wilcoxon): {n_train_subj}"
    )
    lines.append(
        f"- FDR<0.05 within family: **{len(fdr_sig)}**"
    )
    lines.append(
        f"- FDR<0.05 globally (all features, single family): **{len(fdr_sig_global)}**"
    )
    lines.append("")
    lines.append("## Family breakdown")
    lines.append("")
    lines.append("| family | n features | n FDR<0.05 | min p | median |delta| |")
    lines.append("|--------|-----------:|-----------:|------:|---------------:|")
    for _, r in fam_sum.iterrows():
        lines.append(
            f"| {r['family']} | {int(r['n'])} | {int(r['n_sig_fdr'])} | "
            f"{r['min_p']:.2e} | {r['median_abs_delta']:.3f} |"
        )
    lines.append("")
    lines.append("## Top 15 features by raw Wilcoxon p")
    lines.append("")
    lines.append(
        "| rank | feature | family | W | p | p_fdr (family) | Cliff's d | sign | dir | val-dir-pres |"
    )
    lines.append(
        "|-----:|---------|--------|---:|---:|---------------:|---------:|------:|-----|:------------:|"
    )
    for i, (_, r) in enumerate(top15.iterrows(), 1):
        feat = r["feature"]
        vpres = ""
        if val_lookup is not None and feat in val_lookup.index:
            vpres = "yes" if val_lookup.loc[feat, "train_direction_preserved_in_val"] else "no"
        lines.append(
            f"| {i} | `{feat}` | {r['family']} | {r.get('wilcoxon_W', np.nan):.2f} | "
            f"{r['wilcoxon_p']:.2e} | {r['wilcoxon_p_fdr']:.2e} | "
            f"{r['cliffs_delta']:+.3f} | {r['sign_consistency']:.2f} | "
            f"{r['direction']} | {vpres} |"
        )
    lines.append("")
    # Did coupling features beat morphology?
    mo = tests[tests["family"] == "morphology"]
    cpl = tests[tests["family"].isin(["coupling_linear", "coupling_nonlinear"])]
    lines.append("## Did coupling outperform morphology?")
    lines.append("")
    lines.append(
        f"- Morphology: min p = {mo['wilcoxon_p'].min():.2e}, "
        f"median |delta| = {float(np.nanmedian(np.abs(mo['cliffs_delta']))):.3f}"
    )
    lines.append(
        f"- Coupling (linear+nonlinear): min p = {cpl['wilcoxon_p'].min():.2e}, "
        f"median |delta| = {float(np.nanmedian(np.abs(cpl['cliffs_delta']))):.3f}"
    )
    winner = "coupling" if cpl["wilcoxon_p"].min() < mo["wilcoxon_p"].min() else "morphology"
    lines.append(f"- Stronger family (by smallest p): **{winner}**.")
    lines.append("")
    lines.append("## Waveform morphology (notch depth / rise time / aug-index)")
    key_morph = [
        "ppg_notch_depth_rel_mean",
        "ppg_notch_depth_rel_std",
        "ppg_notch_time_frac_mean",
        "ppg_rise_time_mean",
        "ppg_rise_frac_mean",
        "ppg_aug_index_mean",
        "ppg_refl_index_mean",
        "ppg_d2_peak_count_per_beat",
    ]
    lines.append("")
    lines.append("| feature | W | p | p_fdr (family) | delta | direction |")
    lines.append("|---------|---:|---:|--------------:|------:|-----------|")
    for feat in key_morph:
        row = tests[tests["feature"] == feat]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lines.append(
            f"| `{feat}` | {r['wilcoxon_W']:.2f} | {r['wilcoxon_p']:.2e} | "
            f"{r['wilcoxon_p_fdr']:.2e} | {r['cliffs_delta']:+.3f} | {r['direction']} |"
        )
    lines.append("")
    lines.append("## Physiological interpretation")
    lines.append(
        "- **PPG morphology** probes small-vessel compliance and reflected-wave timing. "
        "If PainHand (stimulated at the hand) produces stronger distal vasoconstriction "
        "than PainArm, expect shorter rise times, deeper dicrotic notch relative to "
        "systolic amplitude, and higher augmentation-index proxies on the hand condition."
    )
    lines.append(
        "- **RSA / RESP-BVP phase-locking** indexes parasympathetic modulation. A site-"
        "dependent shift in PLV or mean phase would suggest differential autonomic "
        "routing (afferent input from arm vs hand) even when segment-mean HR is matched."
    )
    lines.append(
        "- **BVP-envelope → EDA lag** captures the delay between pulse-amplitude "
        "responses and sudomotor activation. Hand stimulation might produce a shorter or "
        "larger-magnitude EDA response."
    )
    lines.append(
        "- **MI / dcor** catch nonlinear, non-phase-locked coupling (e.g. saturation, "
        "threshold effects) that linear r misses."
    )
    lines.append("")
    n_pres = (
        int(val_df["train_direction_preserved_in_val"].sum()) if len(val_df) else 0
    )
    lines.append("## Validation reproducibility (top 15)")
    lines.append(
        f"- Direction preserved in validation: **{n_pres}/{len(val_df)}** "
        f"({(100*n_pres/max(len(val_df),1)):.1f}%)"
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append(
        "- Subject-z normalisation removes per-subject level effects; remaining "
        "class differences are shape/coupling-driven, not amplitude-driven."
    )
    lines.append(
        "- 10-s segments are short for respiratory-band PLV; PLV estimates are noisy "
        "per segment but the mean across 12 segments/class/subject is usable."
    )
    lines.append(
        "- FDR is applied within each feature family (morphology / linear coupling / "
        "nonlinear coupling) and also globally. Family-wise is the primary hypothesis."
    )
    lines.append(
        "- SpO2-involved pairs are skipped for constant-SpO2 segments (see inventory)."
    )
    (REPORTS_DIR / "10_morphology_coupling_summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[1/6] Building feature matrix...")
    feats_df = build_feature_matrix()
    feature_cols = [c for c in feats_df.columns if c not in META_COLS]
    print(f"  {len(feats_df)} segments × {len(feature_cols)} features")

    parquet_fp = TABLES_DIR / "morphology_coupling_features.parquet"
    feats_df.to_parquet(parquet_fp, index=False)
    print(f"  wrote {parquet_fp}")

    print("[2/6] Running ARM-vs-HAND tests on train subject-means...")
    tests = run_armhand_tests(feats_df, feature_cols)
    tests_fp = TABLES_DIR / "morphology_coupling_tests.csv"
    tests.to_csv(tests_fp, index=False)
    print(f"  wrote {tests_fp}")
    n_fdr = int((tests["wilcoxon_p_fdr"] < 0.05).sum())
    print(f"  n features FDR<0.05 (family-wise): {n_fdr}")

    print("[3/6] Validation reproducibility (top 15)...")
    top15 = tests.sort_values("wilcoxon_p").head(15)["feature"].tolist()
    val_df = validation_reproducibility(feats_df, top15)
    val_df.to_csv(TABLES_DIR / "morphology_coupling_val_repro.csv", index=False)

    print("[4/6] Loading train tensor for plots...")
    tensor_tr, meta_tr = load_split("train")
    tensor_tr = subject_z_normalise(tensor_tr, meta_tr)

    print("[5/6] Plotting...")
    try:
        plot_ppg_beat_template(feats_df, tensor_tr, meta_tr)
    except Exception as e:
        print(f"  plot_ppg_beat_template failed: {e}")
    try:
        plot_resp_bvp_phase_hist(feats_df)
    except Exception as e:
        print(f"  plot_resp_bvp_phase_hist failed: {e}")
    try:
        plot_lagged_xcorr_bvp_eda(tensor_tr, meta_tr)
    except Exception as e:
        print(f"  plot_lagged_xcorr_bvp_eda failed: {e}")
    try:
        plot_sync_index_violin(feats_df)
    except Exception as e:
        print(f"  plot_sync_index_violin failed: {e}")
    try:
        plot_mi_heatmap(feats_df)
    except Exception as e:
        print(f"  plot_mi_heatmap failed: {e}")
    try:
        plot_top20_wilcoxon_bar(tests)
    except Exception as e:
        print(f"  plot_top20_wilcoxon_bar failed: {e}")

    print("[6/6] Writing report...")
    write_report(
        tests,
        val_df,
        n_features=len(feature_cols),
        n_train_subj=int(feats_df[feats_df["split"] == "train"]["subject"].nunique()),
    )
    print("Done.")


if __name__ == "__main__":
    main()
