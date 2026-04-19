"""
AI4Pain 2026 dataset loader.

Directory layout (per split):
    Dataset/{train,validation}/{Bvp,Eda,Resp,SpO2}/{subject_id}.csv

Each CSV has 36 columns named like:
    {sid}_Baseline_{k}   for k in 1..12   (No-Pain class)
    {sid}_ARM_{k}        for k in 1..12   (Pain-Arm class)
    {sid}_HAND_{k}       for k in 1..12   (Pain-Hand class)

Sampling rate: 100 Hz. Nominal segment length: 10 s -> ~1000 samples,
actual files carry ~1118 rows per segment (first row is header).

Primary API:
    load_split(split)      -> (tensor, meta_df)
        tensor : (n_segments, n_signals, n_samples) float32, NaN-padded
        meta_df: one row per segment with columns
                 split, subject, class, segment_idx, segment_id,
                 and raw_len_{signal} for the pre-truncation lengths.

    SIGNALS, CLASSES, SFREQ, DEFAULT_N_SAMPLES are exported constants.

Cache: arrays -> cache/{split}_tensor.npz
       meta   -> cache/{split}_meta.parquet
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS = ("Bvp", "Eda", "Resp", "SpO2")
CLASSES = ("NoPain", "PainArm", "PainHand")
SFREQ = 100  # Hz
DEFAULT_N_SAMPLES = 1000  # 10 s at 100 Hz (truncate/pad target)


# ---------------------------------------------------------------------------
# Column-name parsing
# ---------------------------------------------------------------------------
_CLASS_ALIASES = {
    "baseline": "NoPain",
    "nopain": "NoPain",
    "no_pain": "NoPain",
    "arm": "PainArm",
    "painarm": "PainArm",
    "pain_arm": "PainArm",
    "hand": "PainHand",
    "painhand": "PainHand",
    "pain_hand": "PainHand",
}


def parse_column(col: str) -> tuple[int, str, int]:
    """Parse '<sid>_<tag>_<idx>' -> (subject_id, class_name, segment_idx).

    The class label is canonicalised to one of CLASSES.
    Segment index is 1-based as in the files.
    """
    parts = col.strip().split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected column name {col!r}")
    sid = int(parts[0])
    idx = int(parts[-1])
    tag = "_".join(parts[1:-1]).lower()
    if tag not in _CLASS_ALIASES:
        raise ValueError(f"Unknown class tag {tag!r} in column {col!r}")
    return sid, _CLASS_ALIASES[tag], idx


# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetPaths:
    """Locates the AI4Pain 2026 raw CSVs.

    Two layouts are accepted under `root`:
      A. Local repo:   <root>/Dataset/{train,validation}/<Signal>/<sid>.csv
      B. Kaggle input: <root>/{train,validation}/<Signal>/<sid>.csv  (no ``Dataset`` infix)

    The first layout is detected first; the second is used as fallback.
    """

    root: Path

    def split_dir(self, split: str) -> Path:
        a = self.root / "Dataset" / split
        if a.exists():
            return a
        return self.root / split

    def signal_dir(self, split: str, signal: str) -> Path:
        return self.split_dir(split) / signal

    def subjects(self, split: str, signal: str = "Bvp") -> list[int]:
        d = self.signal_dir(split, signal)
        return sorted(int(p.stem) for p in d.glob("*.csv"))


import os as _os  # noqa: E402  (kept local to limit scope)

DEFAULT_ROOT = Path(
    _os.environ.get("AI4PAIN_ROOT", "/stuff/Study/projects/AI4Pain 2026 Dataset")
)


def default_paths() -> DatasetPaths:
    return DatasetPaths(DEFAULT_ROOT)


def cache_dir(paths: DatasetPaths | None = None) -> Path:
    """Where cached tensors / metadata go.

    Resolution order:
      1. ``$AI4PAIN_CACHE``           — explicit override
      2. ``/kaggle/working/cache``    — when ``root`` is a read-only Kaggle input
      3. ``<root>/cache``             — local default
    """
    paths = paths or default_paths()
    explicit = _os.environ.get("AI4PAIN_CACHE")
    if explicit:
        d = Path(explicit)
    elif str(paths.root).startswith("/kaggle/input"):
        d = Path("/kaggle/working/cache")
    else:
        d = paths.root / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Core per-file loaders
# ---------------------------------------------------------------------------
def load_subject_signal(
    paths: DatasetPaths, split: str, subject: int, signal: str
) -> pd.DataFrame:
    """Load a single (subject, signal) CSV. Returns a DataFrame
    with rows = time samples, columns = 36 segment names."""
    fp = paths.signal_dir(split, signal) / f"{subject}.csv"
    return pd.read_csv(fp)


# ---------------------------------------------------------------------------
# Split-level tensor builder
# ---------------------------------------------------------------------------
def _build_split(
    paths: DatasetPaths, split: str, n_samples: int
) -> tuple[np.ndarray, pd.DataFrame]:
    subjects = paths.subjects(split)

    meta_rows: list[dict] = []
    arrays: list[np.ndarray] = []  # each (n_signals, n_samples) float32

    for sid in subjects:
        sig_data = {
            sig: load_subject_signal(paths, split, sid, sig) for sig in SIGNALS
        }
        cols = sig_data["Bvp"].columns
        # Sanity: every signal must have the same 36 segment columns
        for sig in SIGNALS:
            if list(sig_data[sig].columns) != list(cols):
                raise RuntimeError(
                    f"Column mismatch for subject {sid} signal {sig}"
                )
        for col in cols:
            subj, cls, idx = parse_column(col)
            assert subj == sid
            raw_lens: dict[str, int] = {}
            stacked = np.full((len(SIGNALS), n_samples), np.nan, dtype=np.float32)
            for s_i, sig in enumerate(SIGNALS):
                arr = sig_data[sig][col].to_numpy(dtype=np.float32)
                raw_lens[sig] = int(arr.shape[0])
                k = min(n_samples, arr.shape[0])
                stacked[s_i, :k] = arr[:k]
            arrays.append(stacked)
            meta_rows.append(
                {
                    "split": split,
                    "subject": sid,
                    "class": cls,
                    "segment_idx": idx,
                    "segment_id": f"{sid}_{cls}_{idx}",
                    **{f"raw_len_{sig}": raw_lens[sig] for sig in SIGNALS},
                }
            )

    tensor = np.stack(arrays, axis=0)  # (N, 4, n_samples)
    meta = pd.DataFrame(meta_rows)
    return tensor, meta


def load_split(
    split: str = "train",
    paths: DatasetPaths | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    cache: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load all segments of a split.

    Returns (tensor, meta_df):
        tensor: (N, 4, n_samples) float32 (NaN-padded if short)
        meta_df: per-segment metadata DataFrame
    """
    paths = paths or default_paths()
    cdir = cache_dir(paths)
    npz_fp = cdir / f"{split}_tensor_{n_samples}.npz"
    meta_fp = cdir / f"{split}_meta_{n_samples}.parquet"
    if cache and npz_fp.exists() and meta_fp.exists():
        with np.load(npz_fp) as z:
            tensor = z["tensor"]
        meta = pd.read_parquet(meta_fp)
        return tensor, meta

    tensor, meta = _build_split(paths, split, n_samples)
    if cache:
        np.savez_compressed(npz_fp, tensor=tensor)
        meta.to_parquet(meta_fp, index=False)
    return tensor, meta


# ---------------------------------------------------------------------------
# Class-label helpers
# ---------------------------------------------------------------------------
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}


def class_codes(meta: pd.DataFrame) -> np.ndarray:
    return meta["class"].map(CLASS_TO_INT).to_numpy()


def pain_binary(meta: pd.DataFrame) -> np.ndarray:
    """0 = NoPain, 1 = any pain (Arm or Hand)."""
    return (meta["class"] != "NoPain").astype(int).to_numpy()


def signal_index(signal: str) -> int:
    return SIGNALS.index(signal)
