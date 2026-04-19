"""Tier-C (Paper 2 idea): Pulse Decomposition Analysis on BVP.

Physiological motivation
------------------------
Each finger BVP pulse can be decomposed into 3 Gaussians:
   G1 = forward systolic wave (ejection)
   G2 = first reflected wave (renal bifurcation / upper-body reflection)
   G3 = second reflected wave (iliac bifurcation / lower-body reflection)

TENS on the **inner forearm** vs **back of hand** alters the cutaneous
vascular bed at different positions along the arterial tree. Expectation:
reflection *timing* (mu2-mu1, mu3-mu1) and *amplitude ratios* (A2/A1,
A3/A1) may differ more between sites than the gross PPG amplitude that
earlier kinetics features already failed to separate.

What this script does
- Per pulse (onset->next onset): fit 3-Gaussian model via least_squares
- Aggregate per segment: median + IQR of {A_i, mu_i-mu_1, sigma_i, ratios}
- Wilcoxon paired (ARM vs HAND) on subject-means + BH-FDR
- LOSO LogReg (subject-z) on train, macro-F1 ARM vs HAND
- Validation split macro-F1
- Writes parquet + markdown report
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as spsig
from scipy import stats as spstats
from scipy.optimize import least_squares
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
RNG = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
REP = ROOT / "results" / "reports"
TAB.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

BVP_IDX = SIGNALS.index("Bvp")


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _strip_nan(x: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(x)
    if not mask.any():
        return np.array([], dtype=np.float32)
    last = int(np.flatnonzero(mask)[-1]) + 1
    return x[:last].astype(np.float32)


def _bandpass(x: np.ndarray, lo=0.5, hi=8.0, fs=SFREQ, order=3) -> np.ndarray:
    if len(x) < 3 * order + 1:
        return x
    try:
        sos = spsig.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
        return spsig.sosfiltfilt(sos, x)
    except Exception:
        return x


def _find_pulse_onsets(x: np.ndarray, fs: float) -> np.ndarray:
    """Return onset indices = local minima between systolic peaks."""
    min_dist = int(0.4 * fs)
    sd = float(np.std(x))
    prom = 0.3 * sd if sd > 0 else None
    peaks, _ = spsig.find_peaks(x, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return np.array([], dtype=int)
    onsets = [int(np.argmin(x[:peaks[0]])) if peaks[0] > 0 else 0]
    for i in range(len(peaks) - 1):
        seg_lo, seg_hi = peaks[i], peaks[i + 1]
        onsets.append(int(seg_lo + np.argmin(x[seg_lo:seg_hi])))
    return np.array(onsets, dtype=int)


# ---------------------------------------------------------------------------
# 3-Gaussian PDA fit
# ---------------------------------------------------------------------------
def _three_gauss(t, A1, m1, s1, A2, m2, s2, A3, m3, s3):
    return (A1 * np.exp(-0.5 * ((t - m1) / s1) ** 2)
            + A2 * np.exp(-0.5 * ((t - m2) / s2) ** 2)
            + A3 * np.exp(-0.5 * ((t - m3) / s3) ** 2))


def _fit_pulse(y: np.ndarray, fs: float) -> dict[str, float] | None:
    n = len(y)
    if n < 10:
        return None
    y = y - y[0]
    mx = float(np.max(y))
    if mx <= 0:
        return None
    t = np.arange(n) / fs
    T = t[-1]
    p0 = [mx, 0.2 * T, 0.08 * T,
          0.4 * mx, 0.5 * T, 0.10 * T,
          0.2 * mx, 0.80 * T, 0.12 * T]
    lb = [0.0, 0.0, 0.01,
          0.0, 0.1 * T, 0.01,
          0.0, 0.3 * T, 0.01]
    ub = [3 * mx + 1e-6, 0.5 * T, 0.4 * T,
          2 * mx + 1e-6, 0.8 * T, 0.4 * T,
          2 * mx + 1e-6, T, 0.5 * T]

    def resid(p):
        return _three_gauss(t, *p) - y

    try:
        r = least_squares(resid, p0, bounds=(lb, ub), max_nfev=200, method="trf")
    except Exception:
        return None
    if not r.success:
        return None
    A1, m1, s1, A2, m2, s2, A3, m3, s3 = r.x
    # Sort components by mu to canonicalise (guarantees m1 <= m2 <= m3)
    comps = sorted([(A1, m1, s1), (A2, m2, s2), (A3, m3, s3)], key=lambda c: c[1])
    (A1, m1, s1), (A2, m2, s2), (A3, m3, s3) = comps
    if A1 <= 1e-6:
        return None
    return {
        "A1": A1, "A2": A2, "A3": A3,
        "mu1": m1, "mu2": m2, "mu3": m3,
        "sig1": s1, "sig2": s2, "sig3": s3,
        "dmu21": m2 - m1, "dmu31": m3 - m1,
        "r21": A2 / A1, "r31": A3 / A1,
        "rmse": float(np.sqrt(np.mean(r.fun ** 2))) / (mx + 1e-9),
    }


FEAT_KEYS = ["A1", "A2", "A3", "mu1", "mu2", "mu3", "sig1", "sig2", "sig3",
             "dmu21", "dmu31", "r21", "r31", "rmse"]
FEATURE_NAMES = [f"pda_{k}_{stat}" for k in FEAT_KEYS for stat in ("median", "iqr")]


def extract_pda(bvp_raw: np.ndarray, fs: float = SFREQ) -> dict[str, float]:
    feats = {f: np.nan for f in FEATURE_NAMES}
    bvp = _strip_nan(bvp_raw)
    if len(bvp) < int(2 * fs):
        return feats
    x = _bandpass(bvp, 0.5, 8.0, fs)
    if not np.isfinite(x).all() or np.std(x) <= 0:
        return feats
    onsets = _find_pulse_onsets(x, fs)
    if len(onsets) < 3:
        return feats
    pulse_feats: list[dict[str, float]] = []
    for i in range(len(onsets) - 1):
        y = x[onsets[i]:onsets[i + 1]]
        if len(y) < int(0.3 * fs):
            continue
        f = _fit_pulse(y.astype(float), fs)
        if f is not None and f["rmse"] < 0.5:  # reject bad fits
            pulse_feats.append(f)
    if len(pulse_feats) < 2:
        return feats
    arr = {k: np.array([pf[k] for pf in pulse_feats], dtype=float) for k in FEAT_KEYS}
    for k in FEAT_KEYS:
        v = arr[k]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        feats[f"pda_{k}_median"] = float(np.median(v))
        q75, q25 = np.percentile(v, [75, 25])
        feats[f"pda_{k}_iqr"] = float(q75 - q25)
    return feats


# ---------------------------------------------------------------------------
# Build feature table
# ---------------------------------------------------------------------------
def build_features() -> pd.DataFrame:
    rows: list[dict] = []
    for split in ("train", "validation"):
        tensor, meta = load_split(split)
        for i in tqdm(range(len(tensor)), desc=f"PDA {split}", ncols=88):
            base = {c: meta.iloc[i][c] for c in
                    ["split", "subject", "class", "segment_idx", "segment_id"]}
            try:
                f = extract_pda(tensor[i, BVP_IDX], fs=SFREQ)
            except Exception:
                f = {k: np.nan for k in FEATURE_NAMES}
            base.update(f)
            rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stats + classification
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
        p = clf.predict(X[te_i])
        scores.append(f1_score(y[te_i], p, average="macro"))
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
    print("[22] extracting PDA features...")
    df = build_features()
    df.to_parquet(TAB / "tierC_pda_bvp_features.parquet", index=False)
    print(f"[22] wrote {TAB / 'tierC_pda_bvp_features.parquet'}  ({len(df)} rows, "
          f"{len(FEATURE_NAMES)} features)")

    print("[22] paired Wilcoxon (ARM vs HAND, subject means)...")
    tests = paired_wilcoxon_armhand(df)
    tests.to_csv(TAB / "tierC_pda_bvp_tests.csv", index=False)
    n_sig = int((tests["p_fdr"] < 0.05).sum())

    print("[22] LOSO ARM vs HAND (LogReg + subject-z)...")
    f1_mean, f1_std, _ = loso_armhand(df)
    f1_val = val_armhand(df)

    with open(REP / "tierC_pda_bvp_summary.md", "w") as fp:
        fp.write("# Tier-C PDA on BVP — ARM vs HAND\n\n")
        fp.write(f"- features: {len(FEATURE_NAMES)}\n")
        fp.write(f"- segments: {len(df)}\n")
        fp.write(f"- FDR<0.05 survivors: **{n_sig} / {len(tests)}**\n")
        fp.write(f"- LOSO macro-F1: **{f1_mean:.3f} ± {f1_std:.3f}** "
                 f"(chance = 0.50)\n")
        fp.write(f"- Validation macro-F1: **{f1_val:.3f}**\n\n")
        fp.write("## Top-10 features by p-value\n\n")
        fp.write(tests.head(10).to_markdown(index=False))
        fp.write("\n")
    print(f"[22] wrote {REP / 'tierC_pda_bvp_summary.md'}")
    print(f"[22] LOSO macro-F1 = {f1_mean:.3f} ± {f1_std:.3f} | "
          f"val = {f1_val:.3f} | FDR<0.05: {n_sig}")


if __name__ == "__main__":
    main()
