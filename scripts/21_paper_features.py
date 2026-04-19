"""Tier-C paper-derived feature extraction.

Adapts ideas from AI4Pain 2025 papers to ARM-vs-HAND localisation:

  1. Wavelet decomposition (Paper 3) — discrete wavelet (db1) of each
     signal at 3 narrow bands; aggregate {mean, median, std} per band.
     RESP is our winning channel; multi-scale RESP shape may carry
     incremental signal beyond aggregate stats.

  2. Arc length (Paper 2 top BVP feature) — sum(|diff(x)|). Captures
     waveform complexity that mean / std miss.

  3. Fuzzy entropy (Paper 2) — non-linear regularity measure, more
     stable than sample entropy on short windows.

Output: results/tables/tierC_paper_features.parquet
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pywt  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data_loader import SIGNALS, load_split  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def strip_nans(x: np.ndarray) -> np.ndarray:
    m = ~np.isnan(x)
    if not m.any():
        return np.array([], dtype=np.float32)
    last = np.nonzero(m)[0][-1] + 1
    return x[:last].astype(np.float64, copy=False)


def arc_length(x: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.sum(np.abs(np.diff(x))))


def fuzzy_entropy(x: np.ndarray, m: int = 2, r: float | None = None,
                  n: float = 2.0) -> float:
    """Fuzzy entropy with Gaussian-like exponential membership.
    Sub-samples to keep O(N^2) tractable on long segments."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < m + 2:
        return float("nan")
    if r is None:
        sd = np.std(x)
        if sd <= 0:
            return float("nan")
        r = 0.2 * sd
    # subsample for speed
    if x.size > 500:
        x = x[::2]
    N = x.size

    def _phi(mm: int) -> float:
        templates = np.array([x[i:i + mm] - np.mean(x[i:i + mm])
                              for i in range(N - mm + 1)])
        # Chebyshev distance
        d = np.max(np.abs(templates[:, None, :] - templates[None, :, :]),
                   axis=2)
        # exclude self
        np.fill_diagonal(d, np.inf)
        # fuzzy similarity (exp(-d^n / r))
        sim = np.exp(-(d ** n) / r)
        return float(np.sum(sim) / ((N - mm + 1) * (N - mm)))

    try:
        B = _phi(m)
        A = _phi(m + 1)
        if B <= 0 or A <= 0:
            return float("nan")
        return float(-np.log(A / B))
    except Exception:
        return float("nan")


def wavelet_band_features(x: np.ndarray, signal: str) -> dict[str, float]:
    """Discrete wavelet decomposition (db1, level 6) -> band stats."""
    out: dict[str, float] = {}
    if x.size < 64:
        return out
    try:
        coeffs = pywt.wavedec(x, "db1", level=6)
    except Exception:
        return out
    # cA6, cD6, cD5, cD4, cD3, cD2, cD1
    # frequencies (assuming fs=100): cD1 25-50Hz, cD2 12.5-25, cD3 6.25-12.5,
    # cD4 3.13-6.25, cD5 1.56-3.13, cD6 0.78-1.56, cA6 0-0.78
    # For RESP/EDA signals of interest: cA6 (slow, 0-0.78Hz), cD6 (0.78-1.56),
    # cD5 (1.56-3.13).
    band_map = {
        "low": coeffs[0],   # 0-0.78 Hz
        "mid": coeffs[1],   # 0.78-1.56 Hz
        "high": coeffs[2],  # 1.56-3.13 Hz
    }
    for band, c in band_map.items():
        if c.size == 0:
            continue
        sd = float(np.std(c))
        out[f"{signal}_wavelet_{band}_mean"] = float(np.mean(c))
        out[f"{signal}_wavelet_{band}_median"] = float(np.median(c))
        out[f"{signal}_wavelet_{band}_std"] = sd
        out[f"{signal}_wavelet_{band}_energy"] = float(np.sum(c ** 2))
    return out


# ---------------------------------------------------------------------------
def process_split(split: str) -> pd.DataFrame:
    tensor, meta = load_split(split)
    rows: list[dict] = []
    for i in tqdm(range(len(meta)), desc=f"tierC {split}"):
        row = {
            "split": split,
            "subject": int(meta.iloc[i]["subject"]),
            "class": meta.iloc[i]["class"],
            "segment_idx": int(meta.iloc[i]["segment_idx"]),
            "segment_id": meta.iloc[i]["segment_id"],
        }
        for s_i, sig in enumerate(SIGNALS):
            x = strip_nans(tensor[i, s_i])
            if x.size < 16:
                continue
            row[f"{sig}_arc_length"] = arc_length(x)
            row[f"{sig}_fuzzy_entropy"] = fuzzy_entropy(x)
            row.update(wavelet_band_features(x, sig))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parts = []
    for split in ("train", "validation"):
        parts.append(process_split(split))
    df = pd.concat(parts, ignore_index=True, sort=False)
    fp = TAB / "tierC_paper_features.parquet"
    df.to_parquet(fp, index=False)
    print(f"[save] {fp}  shape={df.shape}")
    feat = [c for c in df.columns
            if c not in ("split", "subject", "class", "segment_idx", "segment_id")]
    print(f"  n features: {len(feat)}")
    print(f"  sample: {feat[:6]}")


if __name__ == "__main__":
    main()
