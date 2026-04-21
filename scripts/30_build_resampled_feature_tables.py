"""Build resampled feature tables for AI4Pain 2026.

This script creates alternate merged feature tables where each segment is first
resampled to a fixed target length instead of being truncated / NaN-padded.

Methods
-------
- linear: piecewise-linear interpolation to target length
- poly: scipy.signal.resample_poly with anti-aliasing FIR filtering

Outputs
-------
- results/tables/all_features_merged_linear1022.parquet
- results/tables/all_features_merged_poly1022.parquet
- results/reports/30_build_resampled_feature_tables.md
- plots/resampling/*.png

Usage
-----
    uv run python scripts/30_build_resampled_feature_tables.py
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

from src.data_loader import SFREQ, SIGNALS, load_split

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
TAB_DIR = ROOT / "results" / "tables"
REPORT_DIR = ROOT / "results" / "reports"
PLOT_DIR = ROOT / "plots" / "resampling"
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]
RAW_N_SAMPLES = 1118
TARGET_N_SAMPLES = 1022


def load_script_module(script_name: str, module_name: str):
    path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resample_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.size == target_len:
        return x.astype(np.float32, copy=False)
    xp = np.arange(x.size, dtype=np.float64)
    xq = np.linspace(0, x.size - 1, target_len, dtype=np.float64)
    return np.interp(xq, xp, x.astype(np.float64)).astype(np.float32)


def resample_poly_safe(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.size == target_len:
        return x.astype(np.float32, copy=False)
    y = resample_poly(x.astype(np.float64), up=target_len, down=x.size, padtype="line")
    if y.size > target_len:
        y = y[:target_len]
    elif y.size < target_len:
        pad_value = float(y[-1]) if y.size else 0.0
        y = np.pad(y, (0, target_len - y.size), mode="constant", constant_values=pad_value)
    return y.astype(np.float32)


def resample_signal(x: np.ndarray, method: str, target_len: int) -> np.ndarray:
    if method == "linear":
        return resample_linear(x, target_len)
    if method == "poly":
        return resample_poly_safe(x, target_len)
    raise ValueError(method)


def extract_tf_features(tf_mod, tensor, meta: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict] = []
    fail_counts: dict[str, int] = {}
    for i in tqdm(range(len(meta)), desc=f"tf {split}", leave=False):
        base = {
            "split": split,
            "subject": int(meta.iloc[i]["subject"]),
            "class": meta.iloc[i]["class"],
            "segment_idx": int(meta.iloc[i]["segment_idx"]),
            "segment_id": meta.iloc[i]["segment_id"],
        }
        feats: dict[str, float] = {}
        for s_i, sig in enumerate(SIGNALS):
            try:
                feats.update(tf_mod.compute_all(tensor[i, s_i], SFREQ, sig))
            except Exception:
                fail_counts[sig] = fail_counts.get(sig, 0) + 1
        rows.append({**base, **feats})
    return pd.DataFrame(rows), fail_counts


def merge_feature_tables(physio: pd.DataFrame, tf: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    physio_feats = [c for c in physio.columns if c not in META_COLS]
    tf_feats = [c for c in tf.columns if c not in META_COLS]
    raw_feats = [c for c in raw.columns if c not in META_COLS]
    collisions = (set(raw_feats) & set(physio_feats)) | (set(raw_feats) & set(tf_feats))
    rename_map = {c: f"raw_{c}" for c in collisions}
    raw_renamed = raw.rename(columns=rename_map)
    merged = physio.merge(
        tf.drop(columns=[c for c in META_COLS if c != "segment_id"]),
        on="segment_id",
        how="inner",
    )
    merged = merged.merge(
        raw_renamed.drop(columns=[c for c in META_COLS if c != "segment_id"]),
        on="segment_id",
        how="inner",
    )
    return merged, rename_map


def build_resampled_tensor(split: str, method: str, target_len: int) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    tensor_raw, meta = load_split(split, n_samples=RAW_N_SAMPLES)
    out = np.zeros((len(meta), len(SIGNALS), target_len), dtype=np.float32)
    lengths = np.zeros((len(meta), len(SIGNALS)), dtype=np.int32)
    for i in tqdm(range(len(meta)), desc=f"resample {split} {method}"):
        for s_i, sig in enumerate(SIGNALS):
            raw_len = int(meta.iloc[i][f"raw_len_{sig}"])
            x = tensor_raw[i, s_i, :raw_len]
            lengths[i, s_i] = raw_len
            out[i, s_i] = resample_signal(x, method=method, target_len=target_len)
    return out, meta, lengths


def save_example_plots(split: str, tensor_raw: np.ndarray, tensor_map: dict[str, np.ndarray], meta: pd.DataFrame, lengths: np.ndarray) -> list[Path]:
    out_paths: list[Path] = []
    want_lengths = [990, 1022, 1118]
    picks: list[int] = []
    for wl in want_lengths:
        idx = np.where(lengths[:, SIGNALS.index("Resp")] == wl)[0]
        if len(idx):
            picks.append(int(idx[0]))
    if not picks:
        picks = [0, min(1, len(meta) - 1), min(2, len(meta) - 1)]
    picks = picks[:3]

    for idx in picks:
        subject = int(meta.iloc[idx]["subject"])
        cls = meta.iloc[idx]["class"]
        seg = int(meta.iloc[idx]["segment_idx"])
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        for ax, sig in zip(axes, ("Resp", "Bvp")):
            s_i = SIGNALS.index(sig)
            raw_len = int(lengths[idx, s_i])
            raw = tensor_raw[idx, s_i, :raw_len]
            ax.plot(np.arange(raw_len), raw, label=f"raw_{raw_len}", linewidth=1.4, color="black")
            trunc = raw[:TARGET_N_SAMPLES] if raw_len >= TARGET_N_SAMPLES else np.pad(raw, (0, TARGET_N_SAMPLES - raw_len), constant_values=np.nan)
            ax.plot(np.arange(TARGET_N_SAMPLES), trunc, label="truncate/pad_1022", alpha=0.7)
            for method, tensor in tensor_map.items():
                ax.plot(np.arange(TARGET_N_SAMPLES), tensor[idx, s_i], label=f"{method}_1022", alpha=0.8)
            ax.set_title(f"{sig} | subj {subject} | {cls} seg {seg} | raw_len={raw_len}")
            ax.set_ylabel(sig)
            ax.legend(loc="best", fontsize=8)
        axes[-1].set_xlabel("sample")
        fig.tight_layout()
        fp = PLOT_DIR / f"resample_example_{split}_{subject}_{cls}_{seg}.png"
        fig.savefig(fp, dpi=140)
        plt.close(fig)
        out_paths.append(fp)
    return out_paths


def tagged_path(base_name: str, tag: str, directory: Path) -> Path:
    stem, ext = base_name.rsplit(".", 1)
    return directory / f"{stem}_{tag}.{ext}"


def write_report(rows: list[dict], example_paths: list[Path], elapsed_s: float, methods: list[str]) -> None:
    lines = ["# 30 — Resampled feature tables\n"]
    lines.append(f"- runtime: {elapsed_s:.1f}s")
    lines.append(f"- target length: {TARGET_N_SAMPLES}")
    lines.append(f"- source length for cache loading: {RAW_N_SAMPLES}")
    lines.append(f"- methods: {', '.join(methods)}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("| method | split | merged shape | tf failures | merged parquet |")
    lines.append("|---|---|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['split']} | {row['merged_shape']} | {row['tf_failures']} | `{row['merged_fp']}` |"
        )
    lines.append("")
    lines.append("## Example Plots")
    lines.append("")
    for fp in example_paths:
        lines.append(f"- `{fp}`")
    (REPORT_DIR / "30_build_resampled_feature_tables.md").write_text("\n".join(lines))


def main(methods: list[str] | None = None, target_len: int = TARGET_N_SAMPLES) -> None:
    t0 = time.time()
    methods = methods or ["linear", "poly"]
    raw_mod = load_script_module("02_raw_stats.py", "ai4pain_raw_stats_resample")
    physio_mod = load_script_module("03_physio_features.py", "ai4pain_physio_resample")
    tf_mod = load_script_module("04_tfdomain_features.py", "ai4pain_tf_resample")

    report_rows: list[dict] = []
    example_paths: list[Path] = []

    for method in methods:
        tag = f"{method}{target_len}"
        raw_frames: list[pd.DataFrame] = []
        physio_frames: list[pd.DataFrame] = []
        tf_frames: list[pd.DataFrame] = []
        tf_fail_counts: dict[str, int] = {}
        tensor_map_for_plot: dict[str, np.ndarray] = {}
        raw_plot_tensor = None
        plot_meta = None
        plot_lengths = None

        for split in ("train", "validation"):
            print(f"[build] {method} | {split}")
            tensor_res, meta, lengths = build_resampled_tensor(split, method=method, target_len=target_len)
            if split == "train":
                tensor_raw, _ = load_split(split, n_samples=RAW_N_SAMPLES)
                raw_plot_tensor = tensor_raw
                plot_meta = meta.copy()
                plot_lengths = lengths.copy()
                tensor_map_for_plot[method] = tensor_res.copy()

            raw_frames.append(raw_mod.per_segment_stats(tensor_res, meta))
            physio_frames.append(physio_mod.extract_all(tensor_res, meta))
            tf_df, fails = extract_tf_features(tf_mod, tensor_res, meta, split)
            tf_frames.append(tf_df)
            tf_fail_counts[split] = sum(fails.values()) if fails else 0

        raw_df = pd.concat(raw_frames, ignore_index=True)
        physio_df = pd.concat(physio_frames, ignore_index=True)
        tf_df = pd.concat(tf_frames, ignore_index=True)
        merged_df, _ = merge_feature_tables(physio_df, tf_df, raw_df)

        merged_fp = tagged_path("all_features_merged.parquet", tag, TAB_DIR)
        raw_fp = tagged_path("raw_stats_per_segment.parquet", tag, TAB_DIR)
        physio_fp = tagged_path("physio_features.parquet", tag, TAB_DIR)
        tf_fp = tagged_path("tf_features.parquet", tag, TAB_DIR)

        raw_df.to_parquet(raw_fp, index=False)
        physio_df.to_parquet(physio_fp, index=False)
        tf_df.to_parquet(tf_fp, index=False)
        merged_df.to_parquet(merged_fp, index=False)

        for split in ("train", "validation"):
            report_rows.append(
                {
                    "method": method,
                    "split": split,
                    "merged_shape": str(tuple(merged_df[merged_df["split"] == split].shape)),
                    "tf_failures": tf_fail_counts[split],
                    "merged_fp": str(merged_fp),
                }
            )

        if raw_plot_tensor is not None and plot_meta is not None and plot_lengths is not None:
            example_paths.extend(save_example_plots("train", raw_plot_tensor, tensor_map_for_plot, plot_meta, plot_lengths))

    write_report(report_rows, example_paths, time.time() - t0, methods=methods)
    print("\n[done] wrote resampled feature tables")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=["linear", "poly"], choices=["linear", "poly"])
    parser.add_argument("--target-len", type=int, default=TARGET_N_SAMPLES)
    args = parser.parse_args()
    main(methods=args.methods, target_len=args.target_len)
