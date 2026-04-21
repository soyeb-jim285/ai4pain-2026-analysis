"""Plot all signals per subject.

For each subject (train + validation), produce a figure with:
  rows = modalities (Bvp, Eda, Resp, SpO2)
  cols = class (Baseline, ARM, HAND)
Each subplot overlays all trials of that modality/class with the mean highlighted.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/stuff/Study/projects/AI4Pain 2026 Dataset")
DATA = ROOT / "Dataset"
OUT = ROOT / "plots" / "per_subject_signals"
OUT.mkdir(parents=True, exist_ok=True)

MODALITIES = ["Bvp", "Eda", "Resp", "SpO2"]
CLASSES = ["Baseline", "ARM", "HAND"]
SPLITS = ["train", "validation"]
CLASS_COLOR = {"Baseline": "#1f77b4", "ARM": "#d62728", "HAND": "#2ca02c"}


def subject_ids(split):
    d = DATA / split / "Bvp"
    return sorted([int(p.stem) for p in d.glob("*.csv")])


def load_subject(split, sid):
    out = {}
    for m in MODALITIES:
        p = DATA / split / m / f"{sid}.csv"
        out[m] = pd.read_csv(p)
    return out


def class_cols(df, klass):
    return [c for c in df.columns if f"_{klass}_" in c]


def plot_subject(split, sid):
    data = load_subject(split, sid)
    fig, axes = plt.subplots(len(MODALITIES), len(CLASSES),
                             figsize=(15, 10), sharex="col")
    for i, m in enumerate(MODALITIES):
        df = data[m]
        for j, k in enumerate(CLASSES):
            ax = axes[i, j]
            cols = class_cols(df, k)
            if not cols:
                ax.set_visible(False)
                continue
            arr = df[cols].to_numpy()  # (T, n_trials)
            t = np.arange(arr.shape[0])
            ax.plot(t, arr, color=CLASS_COLOR[k], alpha=0.25, lw=0.6)
            ax.plot(t, arr.mean(axis=1), color="black", lw=1.4, label="mean")
            if i == 0:
                ax.set_title(f"{k}  (n={arr.shape[1]})")
            if j == 0:
                ax.set_ylabel(m)
            if i == len(MODALITIES) - 1:
                ax.set_xlabel("sample")
            ax.grid(alpha=0.3)
    fig.suptitle(f"subject {sid}  [{split}]  —  all signals per class",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out_path = OUT / f"{split}_subject_{sid:02d}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    total = 0
    for split in SPLITS:
        sids = subject_ids(split)
        print(f"[{split}] {len(sids)} subjects: {sids}")
        for sid in sids:
            p = plot_subject(split, sid)
            total += 1
            print(f"  saved {p.name}")
    print(f"done. {total} plots -> {OUT}")


if __name__ == "__main__":
    main()
