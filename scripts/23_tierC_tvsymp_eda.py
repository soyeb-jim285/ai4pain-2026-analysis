"""Tier-C (Paper 3 idea): TVSymp latency on EDA for ARM vs HAND.

Physiological motivation
------------------------
TVSymp = time-varying sympathetic index, built from narrowband
(0.08-0.24 Hz) filtered phasic EDA, Hilbert envelope, z-normalised.
Paper 3 (Gkikas 2025) used it as a state indicator for pain *level*.

Here we re-purpose it for *localisation*: arm-stimulus-to-finger-sweat
neural path differs from hand-stimulus-to-finger-sweat path (different
spinal entry level). Latency-of-peak and envelope morphology may
discriminate even if peak amplitude is site-agnostic.

Note on VFCDM: paper 3 uses Variable-Frequency Complex Demodulation.
We approximate with a zero-phase Butterworth narrowband (0.08-0.24 Hz)
on the phasic EDA component. Less selective than full VFCDM but
captures the same band; sufficient for feasibility check.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as spsig
from scipy import stats as spstats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import SFREQ, SIGNALS, load_split  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
REP = ROOT / "results" / "reports"
TAB.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

EDA_IDX = SIGNALS.index("Eda")


# ---------------------------------------------------------------------------
# Signal conditioning
# ---------------------------------------------------------------------------
def _strip_nan(x: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(x)
    if not mask.any():
        return np.array([], dtype=np.float32)
    last = int(np.flatnonzero(mask)[-1]) + 1
    return x[:last].astype(np.float32)


def _butter(x: np.ndarray, lo: float | None, hi: float | None, fs: float,
            order: int = 4) -> np.ndarray:
    try:
        if lo is None and hi is not None:
            sos = spsig.butter(order, hi, btype="low", fs=fs, output="sos")
        elif hi is None and lo is not None:
            sos = spsig.butter(order, lo, btype="high", fs=fs, output="sos")
        else:
            sos = spsig.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
        return spsig.sosfiltfilt(sos, x)
    except Exception:
        return x


def _phasic_eda(eda: np.ndarray, fs: float) -> np.ndarray:
    """Low-pass at 5 Hz, subtract tonic (0.05 Hz low-pass). Paper 3 recipe."""
    if len(eda) < int(2 * fs):
        return eda
    y = _butter(eda, None, 5.0, fs, order=4)
    tonic = _butter(y, None, 0.05, fs, order=2)
    phasic = y - tonic
    # Detrend residual drift
    phasic = phasic - np.mean(phasic)
    return phasic


def _tvsymp(phasic: np.ndarray, fs: float) -> np.ndarray:
    """Narrowband 0.08-0.24 Hz + Hilbert envelope, normalised by its std."""
    if len(phasic) < int(4 * fs):
        return np.full_like(phasic, np.nan, dtype=float)
    band = _butter(phasic, 0.08, 0.24, fs, order=4)
    env = np.abs(spsig.hilbert(band))
    sd = float(np.std(env))
    if sd <= 1e-9:
        return np.zeros_like(env)
    return env / sd


FEATURE_NAMES = [
    "tvsymp_peak_amp",
    "tvsymp_peak_time_s",
    "tvsymp_time_to_half_peak_s",
    "tvsymp_rise_slope",
    "tvsymp_auc",
    "tvsymp_centroid_s",
    "tvsymp_mean",
    "tvsymp_std",
    "tvsymp_early_mean_0_3s",
    "tvsymp_late_mean_7_10s",
    "tvsymp_early_minus_late",
    "tvsymp_first_crossing_mean_s",   # time at which envelope first crosses its own mean
]


def extract_tvsymp(eda_raw: np.ndarray, fs: float = SFREQ) -> dict[str, float]:
    feats = {f: np.nan for f in FEATURE_NAMES}
    eda = _strip_nan(eda_raw)
    if len(eda) < int(4 * fs):
        return feats
    phasic = _phasic_eda(eda, fs)
    env = _tvsymp(phasic, fs)
    if not np.isfinite(env).any():
        return feats
    t = np.arange(len(env)) / fs
    pk = int(np.argmax(env))
    peak_amp = float(env[pk])
    feats["tvsymp_peak_amp"] = peak_amp
    feats["tvsymp_peak_time_s"] = float(t[pk])
    half = 0.5 * peak_amp
    above = np.where(env >= half)[0]
    if len(above) > 0:
        feats["tvsymp_time_to_half_peak_s"] = float(t[above[0]])
        if feats["tvsymp_peak_time_s"] > feats["tvsymp_time_to_half_peak_s"]:
            dt = feats["tvsymp_peak_time_s"] - feats["tvsymp_time_to_half_peak_s"]
            feats["tvsymp_rise_slope"] = (peak_amp - half) / max(dt, 1e-3)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    feats["tvsymp_auc"] = float(trapz(env, t))
    s = float(env.sum())
    if s > 0:
        feats["tvsymp_centroid_s"] = float((env * t).sum() / s)
    feats["tvsymp_mean"] = float(np.mean(env))
    feats["tvsymp_std"] = float(np.std(env))
    mask_e = t <= 3.0
    mask_l = (t >= 7.0) & (t <= 10.0)
    if mask_e.any():
        feats["tvsymp_early_mean_0_3s"] = float(np.mean(env[mask_e]))
    if mask_l.any():
        feats["tvsymp_late_mean_7_10s"] = float(np.mean(env[mask_l]))
    if np.isfinite(feats["tvsymp_early_mean_0_3s"]) and \
            np.isfinite(feats["tvsymp_late_mean_7_10s"]):
        feats["tvsymp_early_minus_late"] = (
            feats["tvsymp_early_mean_0_3s"] - feats["tvsymp_late_mean_7_10s"]
        )
    mu = float(np.mean(env))
    crossings = np.where(env >= mu)[0]
    if len(crossings) > 0:
        feats["tvsymp_first_crossing_mean_s"] = float(t[crossings[0]])
    return feats


# ---------------------------------------------------------------------------
# Build feature table
# ---------------------------------------------------------------------------
def build_features() -> pd.DataFrame:
    rows: list[dict] = []
    for split in ("train", "validation"):
        tensor, meta = load_split(split)
        for i in tqdm(range(len(tensor)), desc=f"TVSymp {split}", ncols=88):
            base = {c: meta.iloc[i][c] for c in
                    ["split", "subject", "class", "segment_idx", "segment_id"]}
            try:
                f = extract_tvsymp(tensor[i, EDA_IDX], fs=SFREQ)
            except Exception:
                f = {k: np.nan for k in FEATURE_NAMES}
            base.update(f)
            rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stats + classification (mirror script 22)
# ---------------------------------------------------------------------------
def paired_wilcoxon_armhand(df: pd.DataFrame) -> pd.DataFrame:
    tr = df[df["split"] == "train"]
    rows = []
    for feat in FEATURE_NAMES:
        arm = tr[tr["class"] == "PainArm"].groupby("subject")[feat].mean()
        hand = tr[tr["class"] == "PainHand"].groupby("subject")[feat].mean()
        m = pd.concat([arm, hand], axis=1, keys=["arm", "hand"]).dropna()
        if len(m) < 5:
            rows.append({"feature": feat, "n": len(m), "p": np.nan,
                         "mean_arm": np.nan, "mean_hand": np.nan})
            continue
        try:
            _, p = spstats.wilcoxon(m["arm"], m["hand"], zero_method="wilcox")
        except Exception:
            p = np.nan
        rows.append({"feature": feat, "n": len(m), "p": float(p),
                     "mean_arm": float(m["arm"].mean()),
                     "mean_hand": float(m["hand"].mean())})
    out = pd.DataFrame(rows)
    mask = out["p"].notna()
    p_fdr = np.full(len(out), np.nan)
    if mask.any():
        _, padj, _, _ = multipletests(out.loc[mask, "p"], alpha=0.05, method="fdr_bh")
        p_fdr[mask.to_numpy()] = padj
    out["p_fdr"] = p_fdr
    return out.sort_values("p")


def _subject_z(X: np.ndarray, subj: np.ndarray) -> np.ndarray:
    out = X.astype(float).copy()
    for s in np.unique(subj):
        m = subj == s
        mu = np.nanmean(out[m], axis=0)
        sd = np.nanstd(out[m], axis=0)
        sd[sd < 1e-9] = 1.0
        out[m] = (out[m] - mu) / sd
    return out


def loso_armhand(df: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    tr = df[(df["split"] == "train") & df["class"].isin(["PainArm", "PainHand"])].copy()
    X = tr[FEATURE_NAMES].to_numpy(dtype=float)
    y = (tr["class"] == "PainHand").astype(int).to_numpy()
    subj = tr["subject"].to_numpy()
    X = _subject_z(X, subj)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    logo = LeaveOneGroupOut()
    scores = []
    for tr_i, te_i in logo.split(X, y, groups=subj):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X[tr_i], y[tr_i])
        scores.append(f1_score(y[te_i], clf.predict(X[te_i]), average="macro"))
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std()), scores


def val_armhand(df: pd.DataFrame) -> float:
    tr = df[(df["split"] == "train") & df["class"].isin(["PainArm", "PainHand"])]
    va = df[(df["split"] == "validation") & df["class"].isin(["PainArm", "PainHand"])]
    if len(va) == 0:
        return float("nan")
    Xt = _subject_z(tr[FEATURE_NAMES].to_numpy(float), tr["subject"].to_numpy())
    Xv = _subject_z(va[FEATURE_NAMES].to_numpy(float), va["subject"].to_numpy())
    imp = SimpleImputer(strategy="median").fit(Xt)
    sca = StandardScaler().fit(imp.transform(Xt))
    Xt = sca.transform(imp.transform(Xt))
    Xv = sca.transform(imp.transform(Xv))
    yt = (tr["class"] == "PainHand").astype(int).to_numpy()
    yv = (va["class"] == "PainHand").astype(int).to_numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xt, yt)
    return float(f1_score(yv, clf.predict(Xv), average="macro"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[23] extracting TVSymp features...")
    df = build_features()
    df.to_parquet(TAB / "tierC_tvsymp_eda_features.parquet", index=False)
    print(f"[23] wrote {TAB / 'tierC_tvsymp_eda_features.parquet'}  "
          f"({len(df)} rows, {len(FEATURE_NAMES)} features)")

    print("[23] paired Wilcoxon (ARM vs HAND, subject means)...")
    tests = paired_wilcoxon_armhand(df)
    tests.to_csv(TAB / "tierC_tvsymp_eda_tests.csv", index=False)
    n_sig = int((tests["p_fdr"] < 0.05).sum())

    print("[23] LOSO ARM vs HAND (LogReg + subject-z)...")
    f1_mean, f1_std, _ = loso_armhand(df)
    f1_val = val_armhand(df)

    with open(REP / "tierC_tvsymp_eda_summary.md", "w") as fp:
        fp.write("# Tier-C TVSymp on EDA — ARM vs HAND\n\n")
        fp.write(f"- features: {len(FEATURE_NAMES)}\n")
        fp.write(f"- segments: {len(df)}\n")
        fp.write(f"- FDR<0.05 survivors: **{n_sig} / {len(tests)}**\n")
        fp.write(f"- LOSO macro-F1: **{f1_mean:.3f} ± {f1_std:.3f}** "
                 f"(chance = 0.50)\n")
        fp.write(f"- Validation macro-F1: **{f1_val:.3f}**\n\n")
        fp.write("Note: narrowband 0.08-0.24 Hz Butterworth approximates VFCDM "
                 "used in paper 3. Less selective; feasibility-check only.\n\n")
        fp.write("## All features by p-value\n\n")
        fp.write(tests.to_markdown(index=False))
        fp.write("\n")
    print(f"[23] wrote {REP / 'tierC_tvsymp_eda_summary.md'}")
    print(f"[23] LOSO macro-F1 = {f1_mean:.3f} ± {f1_std:.3f} | "
          f"val = {f1_val:.3f} | FDR<0.05: {n_sig}")


if __name__ == "__main__":
    main()
