"""AI4Pain 2026 dataset inventory and integrity report.

Computes per-split / per-subject / per-signal summary tables and plots,
saving everything to ``results/tables``, ``results/reports`` and
``plots/inventory``.

Run from the ``analysis`` directory:
    uv run python scripts/01_inventory.py
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (  # noqa: E402
    CLASSES,
    SIGNALS,
    default_paths,
    load_split,
    load_subject_signal,
)

plt.rcParams["figure.dpi"] = 120

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"
REPORTS_DIR = ROOT / "results" / "reports"
PLOTS_DIR = ROOT / "plots" / "inventory"
for d in (TABLES_DIR, REPORTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 1000  # matches loader default
SPLITS = ("train", "validation")


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------
def load_all() -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    out: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for split in SPLITS:
        tensor, meta = load_split(split)
        out[split] = (tensor, meta)
    return out


def scan_raw_csv_nan_inf(split: str) -> pd.DataFrame:
    """Walk every (subject, signal) raw CSV and count NaN/inf per segment.

    Returns a long-form DataFrame with columns:
        split, subject, class, segment_idx, segment_id, signal,
        raw_len, raw_nan_count, raw_inf_count,
        raw_nan_in_first_1000, raw_inf_in_first_1000.
    """
    paths = default_paths()
    rows: list[dict] = []
    subjects = paths.subjects(split)
    for sid in subjects:
        per_sig = {
            sig: load_subject_signal(paths, split, sid, sig) for sig in SIGNALS
        }
        cols = list(per_sig["Bvp"].columns)
        for col in cols:
            # Parse column: {sid}_{tag}_{idx}
            parts = col.split("_")
            idx = int(parts[-1])
            tag = "_".join(parts[1:-1]).lower()
            if tag in ("baseline", "nopain", "no_pain"):
                cls = "NoPain"
            elif tag in ("arm", "painarm", "pain_arm"):
                cls = "PainArm"
            elif tag in ("hand", "painhand", "pain_hand"):
                cls = "PainHand"
            else:
                cls = tag
            for sig in SIGNALS:
                arr = per_sig[sig][col].to_numpy(dtype=np.float32)
                raw_len = arr.shape[0]
                head = arr[: min(N_SAMPLES, raw_len)]
                rows.append(
                    {
                        "split": split,
                        "subject": sid,
                        "class": cls,
                        "segment_idx": idx,
                        "segment_id": f"{sid}_{cls}_{idx}",
                        "signal": sig,
                        "raw_len": raw_len,
                        "raw_nan_count": int(np.isnan(arr).sum()),
                        "raw_inf_count": int(np.isinf(arr).sum()),
                        "raw_nan_in_first_1000": int(np.isnan(head).sum()),
                        "raw_inf_in_first_1000": int(np.isinf(head).sum()),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def build_inventory_subjects(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for split, (_, meta) in splits.items():
        raw_cols = [f"raw_len_{sig}" for sig in SIGNALS]
        for sid, g in meta.groupby("subject"):
            cls_counts = g["class"].value_counts().reindex(CLASSES, fill_value=0)
            all_raw = g[raw_cols].to_numpy().ravel()
            rows.append(
                {
                    "split": split,
                    "subject": int(sid),
                    "n_segments": int(len(g)),
                    "n_NoPain": int(cls_counts["NoPain"]),
                    "n_PainArm": int(cls_counts["PainArm"]),
                    "n_PainHand": int(cls_counts["PainHand"]),
                    "raw_len_min": int(all_raw.min()),
                    "raw_len_mean": float(all_raw.mean()),
                    "raw_len_max": int(all_raw.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "subject"]).reset_index(drop=True)


def build_inventory_segment_lengths(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for split, (_, meta) in splits.items():
        for sig in SIGNALS:
            arr = meta[f"raw_len_{sig}"].to_numpy()
            rows.append(
                {
                    "split": split,
                    "signal": sig,
                    "min": int(arr.min()),
                    "median": float(np.median(arr)),
                    "max": int(arr.max()),
                    "std": float(arr.std()),
                    "n_shorter_than_1000": int((arr < 1000).sum()),
                    "n_longer_than_1000": int((arr > 1000).sum()),
                    "n_equal_1000": int((arr == 1000).sum()),
                    "n_total": int(len(arr)),
                }
            )
    return pd.DataFrame(rows)


def build_inventory_nan_inf(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
    raw_scan: pd.DataFrame,
) -> pd.DataFrame:
    """For each (split, signal), count:

    - raw_nan_segments / raw_inf_segments: any NaN/inf anywhere in raw CSV column.
    - raw_nan_in_first_1000_segments: NaN anywhere in first 1000 samples.
    - tensor_interior_nan_segments: NaN/inf at positions < raw_len (i.e. interior, not padding).
    - tensor_padding_nan_segments: segments where raw_len < 1000 so tail is padded with NaN.
    - tensor_interior_inf_segments: inf anywhere at positions < min(1000, raw_len).
    """
    rows: list[dict] = []
    for split, (tensor, meta) in splits.items():
        for s_i, sig in enumerate(SIGNALS):
            sig_scan = raw_scan[(raw_scan["split"] == split) & (raw_scan["signal"] == sig)]

            interior_nan = 0
            interior_inf = 0
            padding_nan = 0
            for i, row in enumerate(meta.itertuples(index=False)):
                rlen = int(getattr(row, f"raw_len_{sig}"))
                k = min(N_SAMPLES, rlen)
                interior = tensor[i, s_i, :k]
                if np.isnan(interior).any():
                    interior_nan += 1
                if np.isinf(interior).any():
                    interior_inf += 1
                if rlen < N_SAMPLES:
                    padding_nan += 1  # tail of tensor is NaN padding
            rows.append(
                {
                    "split": split,
                    "signal": sig,
                    "raw_any_nan_segments": int((sig_scan["raw_nan_count"] > 0).sum()),
                    "raw_any_inf_segments": int((sig_scan["raw_inf_count"] > 0).sum()),
                    "raw_nan_in_first_1000_segments": int(
                        (sig_scan["raw_nan_in_first_1000"] > 0).sum()
                    ),
                    "raw_inf_in_first_1000_segments": int(
                        (sig_scan["raw_inf_in_first_1000"] > 0).sum()
                    ),
                    "tensor_interior_nan_segments": int(interior_nan),
                    "tensor_interior_inf_segments": int(interior_inf),
                    "tensor_padding_nan_segments": int(padding_nan),
                    "n_total": int(len(meta)),
                }
            )
    return pd.DataFrame(rows)


def build_inventory_constant_segments(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
    std_thresh: float = 1e-8,
) -> pd.DataFrame:
    rows: list[dict] = []
    for split, (tensor, meta) in splits.items():
        for s_i, sig in enumerate(SIGNALS):
            for i, row in enumerate(meta.itertuples(index=False)):
                rlen = int(getattr(row, f"raw_len_{sig}"))
                k = min(N_SAMPLES, rlen)
                seg = tensor[i, s_i, :k]
                seg = seg[np.isfinite(seg)]
                if seg.size < 2:
                    continue
                s = float(seg.std())
                if s < std_thresh:
                    rows.append(
                        {
                            "split": split,
                            "subject": int(row.subject),
                            "class": row._asdict().get("class", None)
                            if hasattr(row, "_asdict")
                            else None,
                            "segment_idx": int(row.segment_idx),
                            "segment_id": row.segment_id,
                            "signal": sig,
                            "raw_len": rlen,
                            "std": s,
                            "mean": float(seg.mean()),
                        }
                    )
    df = pd.DataFrame(rows)
    # The namedtuple approach above can fail for the reserved 'class' attribute;
    # derive it again directly if missing.
    if not df.empty and (df["class"].isna().any() if "class" in df.columns else True):
        # Re-derive class from the segment_id "{sid}_{class}_{idx}"
        df["class"] = df["segment_id"].str.split("_").str[1]
    return df


def build_inventory_duplicate_segments(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
) -> pd.DataFrame:
    """Hash each segment's interior (non-padded) float32 bytes and report any duplicates."""
    rows: list[dict] = []
    for split, (tensor, meta) in splits.items():
        for s_i, sig in enumerate(SIGNALS):
            hash_to_rows: dict[str, list[dict]] = {}
            for i, row in enumerate(meta.itertuples(index=False)):
                rlen = int(getattr(row, f"raw_len_{sig}"))
                k = min(N_SAMPLES, rlen)
                interior = tensor[i, s_i, :k]
                # Use raw bytes of float32 values (ignore padding).
                h = hashlib.sha1(interior.tobytes()).hexdigest()
                hash_to_rows.setdefault(h, []).append(
                    {
                        "split": split,
                        "signal": sig,
                        "subject": int(row.subject),
                        "segment_id": row.segment_id,
                        "segment_idx": int(row.segment_idx),
                        "raw_len": rlen,
                        "hash": h,
                    }
                )
            for h, group in hash_to_rows.items():
                if len(group) > 1:
                    for g in group:
                        g["dup_count"] = len(group)
                        rows.append(g)
    return pd.DataFrame(rows)


def build_inventory_subject_splits(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
) -> tuple[pd.DataFrame, set[int]]:
    per_split: dict[str, set[int]] = {
        split: set(meta["subject"].unique().tolist())
        for split, (_, meta) in splits.items()
    }
    all_subjects = sorted(set().union(*per_split.values()))
    rows: list[dict] = []
    for sid in all_subjects:
        in_splits = [s for s in SPLITS if sid in per_split[s]]
        rows.append(
            {
                "subject": int(sid),
                "in_train": "train" in in_splits,
                "in_validation": "validation" in in_splits,
                "n_splits": len(in_splits),
            }
        )
    overlap = per_split["train"] & per_split["validation"]
    return pd.DataFrame(rows), overlap


def build_inventory_signal_ranges(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for split, (tensor, _meta) in splits.items():
        for s_i, sig in enumerate(SIGNALS):
            data = tensor[:, s_i, :]
            data = data[np.isfinite(data)]
            if data.size == 0:
                continue
            rows.append(
                {
                    "split": split,
                    "signal": sig,
                    "min": float(data.min()),
                    "p1": float(np.percentile(data, 1)),
                    "median": float(np.median(data)),
                    "p99": float(np.percentile(data, 99)),
                    "max": float(data.max()),
                    "n_finite": int(data.size),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_segment_lengths_hist(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
    out_fp: Path,
) -> None:
    fig, axes = plt.subplots(
        len(SIGNALS), len(SPLITS), figsize=(9, 10), sharey="row"
    )
    for i, sig in enumerate(SIGNALS):
        for j, split in enumerate(SPLITS):
            ax = axes[i, j]
            meta = splits[split][1]
            vals = meta[f"raw_len_{sig}"].to_numpy()
            ax.hist(vals, bins=40, color="#4C72B0", edgecolor="white")
            ax.axvline(1000, color="red", linestyle="--", linewidth=1, label="1000")
            ax.set_title(f"{split} — {sig}")
            ax.set_xlabel("raw samples")
            if j == 0:
                ax.set_ylabel("n segments")
            ax.grid(alpha=0.2)
    fig.suptitle("Raw segment length per signal / split", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_fp)
    plt.close(fig)


def plot_subject_class_counts_heatmap(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
    out_fp: Path,
) -> None:
    fig, axes = plt.subplots(
        1, len(SPLITS), figsize=(10, 10),
        gridspec_kw={"width_ratios": [len(splits["train"][1]["subject"].unique()),
                                      len(splits["validation"][1]["subject"].unique())]},
    )
    for j, split in enumerate(SPLITS):
        meta = splits[split][1]
        pivot = (
            meta.groupby(["subject", "class"])
            .size()
            .unstack("class")
            .reindex(columns=list(CLASSES), fill_value=0)
            .sort_index()
        )
        sns.heatmap(
            pivot.T,
            ax=axes[j],
            annot=True,
            fmt="d",
            cmap="viridis",
            cbar=j == 1,
            vmin=0,
            vmax=pivot.values.max(),
        )
        axes[j].set_title(f"{split}")
        axes[j].set_xlabel("subject")
        if j == 0:
            axes[j].set_ylabel("class")
    fig.suptitle("Segments per (subject, class)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fp)
    plt.close(fig)


def plot_signal_range_boxplots(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
    out_fp: Path,
) -> None:
    # Per-segment median per signal per class, across both splits (labelled).
    records: list[dict] = []
    for split, (tensor, meta) in splits.items():
        for s_i, sig in enumerate(SIGNALS):
            for i, row in enumerate(meta.itertuples(index=False)):
                rlen = int(getattr(row, f"raw_len_{sig}"))
                k = min(N_SAMPLES, rlen)
                seg = tensor[i, s_i, :k]
                seg = seg[np.isfinite(seg)]
                if seg.size == 0:
                    continue
                records.append(
                    {
                        "split": split,
                        "signal": sig,
                        "class": row.segment_id.split("_")[1],
                        "median_value": float(np.median(seg)),
                    }
                )
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, sig in zip(axes.ravel(), SIGNALS):
        sub = df[df["signal"] == sig]
        sns.boxplot(
            data=sub,
            x="class",
            y="median_value",
            hue="split",
            order=list(CLASSES),
            ax=ax,
        )
        ax.set_title(sig)
        ax.set_xlabel("class")
        ax.set_ylabel("per-segment median")
        ax.grid(alpha=0.2)
    fig.suptitle("Per-segment median amplitude per signal/class", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_fp)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_summary_report(
    splits: dict[str, tuple[np.ndarray, pd.DataFrame]],
    inv_subjects: pd.DataFrame,
    inv_seglens: pd.DataFrame,
    inv_naninf: pd.DataFrame,
    inv_constant: pd.DataFrame,
    inv_dup: pd.DataFrame,
    inv_subject_splits: pd.DataFrame,
    overlap: set[int],
    inv_ranges: pd.DataFrame,
    runtime_s: float,
) -> str:
    lines: list[str] = []
    lines.append("# AI4Pain 2026 — Inventory & Integrity Summary\n")
    lines.append(f"_Generated in {runtime_s:.1f}s._\n")

    # Shape
    lines.append("## Shape\n")
    for split, (tensor, meta) in splits.items():
        lines.append(
            f"- `{split}`: tensor `{tensor.shape}` ({tensor.dtype}), "
            f"{meta['subject'].nunique()} subjects, {len(meta)} segments."
        )
    lines.append("")

    # Subject/split overlap
    lines.append("## Subject-level")
    if overlap:
        lines.append(
            f"- **WARNING**: {len(overlap)} subjects appear in BOTH splits: "
            f"{sorted(overlap)}"
        )
    else:
        lines.append("- No subject overlap between train and validation (good).")
    # Imbalanced per-subject class counts?
    unbalanced = inv_subjects[
        (inv_subjects["n_NoPain"] != 12)
        | (inv_subjects["n_PainArm"] != 12)
        | (inv_subjects["n_PainHand"] != 12)
    ]
    if unbalanced.empty:
        lines.append("- Every subject has exactly 12/12/12 (NoPain/PainArm/PainHand).")
    else:
        lines.append(f"- {len(unbalanced)} subjects have non-12/12/12 class counts:")
        for _, r in unbalanced.iterrows():
            lines.append(
                f"  - subj {int(r['subject'])} ({r['split']}): "
                f"NoPain={int(r['n_NoPain'])} "
                f"PainArm={int(r['n_PainArm'])} "
                f"PainHand={int(r['n_PainHand'])}"
            )
    lines.append("")

    # Segment lengths
    lines.append("## Segment lengths (pre-truncation)")
    for _, r in inv_seglens.iterrows():
        lines.append(
            f"- `{r['split']}/{r['signal']}`: "
            f"min={int(r['min'])}, median={int(r['median'])}, max={int(r['max'])}, "
            f"std={r['std']:.1f}; "
            f"<1000: {int(r['n_shorter_than_1000'])}, "
            f">1000: {int(r['n_longer_than_1000'])}, "
            f"=1000: {int(r['n_equal_1000'])} / {int(r['n_total'])}"
        )
    # Flag any truncation / padding pressure
    total_short = int(inv_seglens["n_shorter_than_1000"].sum())
    total_long = int(inv_seglens["n_longer_than_1000"].sum())
    lines.append("")
    lines.append(
        f"- Across all (split,signal) pairs: {total_short} cases are SHORTER than 1000 "
        f"(→ tail NaN padding), {total_long} are LONGER (→ samples ≥1000 are dropped)."
    )
    lines.append("")

    # NaN/Inf
    lines.append("## NaN / Inf")
    any_interior = int(
        (
            inv_naninf["tensor_interior_nan_segments"]
            + inv_naninf["tensor_interior_inf_segments"]
        ).sum()
    )
    if any_interior == 0:
        lines.append(
            "- No interior NaN/inf in any tensor segment (padding NaNs accounted for)."
        )
    for _, r in inv_naninf.iterrows():
        if (
            r["tensor_interior_nan_segments"]
            + r["tensor_interior_inf_segments"]
            + r["raw_any_nan_segments"]
            + r["raw_any_inf_segments"]
        ) == 0 and r["tensor_padding_nan_segments"] == 0:
            continue
        lines.append(
            f"- `{r['split']}/{r['signal']}`: "
            f"raw-any-NaN={int(r['raw_any_nan_segments'])}, "
            f"raw-any-inf={int(r['raw_any_inf_segments'])}, "
            f"interior-NaN={int(r['tensor_interior_nan_segments'])}, "
            f"interior-inf={int(r['tensor_interior_inf_segments'])}, "
            f"padding-NaN={int(r['tensor_padding_nan_segments'])} "
            f"/ {int(r['n_total'])}"
        )
    lines.append("")

    # Constant segments
    lines.append("## Constant segments (std < 1e-8)")
    if inv_constant.empty:
        lines.append("- None. No flatlined signals detected.")
    else:
        lines.append(f"- {len(inv_constant)} constant-segment cases:")
        summary = (
            inv_constant.groupby(["split", "signal"]).size().reset_index(name="n")
        )
        for _, r in summary.iterrows():
            lines.append(
                f"  - `{r['split']}/{r['signal']}`: {int(r['n'])} constant segments"
            )
        # Show up to 5 example rows
        lines.append("  - examples:")
        for _, r in inv_constant.head(5).iterrows():
            lines.append(
                f"    - subj {int(r['subject'])} {r['class']} seg {int(r['segment_idx'])} "
                f"({r['signal']}): mean={r['mean']:.4g}, std={r['std']:.2g}"
            )
    lines.append("")

    # Duplicates
    lines.append("## Duplicate signal arrays (byte-identical)")
    if inv_dup.empty:
        lines.append("- None.")
    else:
        dup_summary = (
            inv_dup.groupby(["split", "signal"])
            .size()
            .reset_index(name="n_rows_in_dup_groups")
        )
        for _, r in dup_summary.iterrows():
            lines.append(
                f"- `{r['split']}/{r['signal']}`: "
                f"{int(r['n_rows_in_dup_groups'])} rows in duplicate groups "
                f"(including originals)"
            )
    lines.append("")

    # Signal ranges
    lines.append("## Signal ranges (finite values only)")
    expected = {
        "Bvp": "arbitrary / ~centered",
        "Eda": "µS, typically 0–30 µS",
        "Resp": "arbitrary units",
        "SpO2": "0–100 %",
    }
    for _, r in inv_ranges.iterrows():
        note = expected.get(r["signal"], "")
        flag = ""
        if r["signal"] == "SpO2" and (r["max"] > 100 or r["min"] < 0):
            flag = "  ← out-of-range!"
        if r["signal"] == "Eda" and r["min"] < 0:
            flag = "  ← negative EDA?"
        lines.append(
            f"- `{r['split']}/{r['signal']}` ({note}): "
            f"min={r['min']:.3g}, p1={r['p1']:.3g}, median={r['median']:.3g}, "
            f"p99={r['p99']:.3g}, max={r['max']:.3g}{flag}"
        )
    lines.append("")

    lines.append("## Outputs")
    lines.append("- Tables: `results/tables/inventory_*.csv`")
    lines.append("- Plots:  `plots/inventory/*.png`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    print("Loading splits...")
    splits = load_all()
    for split, (tensor, meta) in splits.items():
        print(f"  {split}: tensor={tensor.shape} meta={len(meta)}")

    print("Scanning raw CSVs for NaN/inf...")
    raw_scan_parts = [scan_raw_csv_nan_inf(split) for split in SPLITS]
    raw_scan = pd.concat(raw_scan_parts, ignore_index=True)

    print("Building tables...")
    inv_subjects = build_inventory_subjects(splits)
    inv_seglens = build_inventory_segment_lengths(splits)
    inv_naninf = build_inventory_nan_inf(splits, raw_scan)
    inv_constant = build_inventory_constant_segments(splits)
    inv_dup = build_inventory_duplicate_segments(splits)
    inv_subj_splits, overlap = build_inventory_subject_splits(splits)
    inv_ranges = build_inventory_signal_ranges(splits)

    inv_subjects.to_csv(TABLES_DIR / "inventory_subjects.csv", index=False)
    inv_seglens.to_csv(TABLES_DIR / "inventory_segment_lengths.csv", index=False)
    inv_naninf.to_csv(TABLES_DIR / "inventory_nan_inf.csv", index=False)
    inv_constant.to_csv(TABLES_DIR / "inventory_constant_segments.csv", index=False)
    inv_dup.to_csv(TABLES_DIR / "inventory_duplicate_segments.csv", index=False)
    inv_subj_splits.to_csv(TABLES_DIR / "inventory_subject_splits.csv", index=False)
    inv_ranges.to_csv(TABLES_DIR / "inventory_signal_ranges.csv", index=False)

    print("Plotting...")
    plot_segment_lengths_hist(splits, PLOTS_DIR / "segment_lengths_hist.png")
    plot_subject_class_counts_heatmap(
        splits, PLOTS_DIR / "subject_class_counts_heatmap.png"
    )
    plot_signal_range_boxplots(splits, PLOTS_DIR / "signal_range_boxplots.png")

    runtime = time.time() - t_start
    print(f"Writing summary report (total runtime so far {runtime:.1f}s)...")
    md = build_summary_report(
        splits,
        inv_subjects,
        inv_seglens,
        inv_naninf,
        inv_constant,
        inv_dup,
        inv_subj_splits,
        overlap,
        inv_ranges,
        runtime,
    )
    (REPORTS_DIR / "01_inventory_summary.md").write_text(md)

    print(f"Done in {time.time() - t_start:.1f}s.")
    print("Outputs:")
    for p in sorted(TABLES_DIR.glob("inventory_*.csv")):
        print(f"  {p}")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {p}")
    print(f"  {REPORTS_DIR / '01_inventory_summary.md'}")


if __name__ == "__main__":
    main()
