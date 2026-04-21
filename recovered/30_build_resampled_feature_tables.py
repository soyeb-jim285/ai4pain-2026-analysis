# Source Generated with Decompyle++
# File: 30_build_resampled_feature_tables.cpython-312.pyc (Python 3.12)

'''Build resampled feature tables for AI4Pain 2026.

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
'''
from __future__ import annotations
import argparse
import importlib.util as importlib
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matplotlib.pyplot import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm
from src.data_loader import SFREQ, SIGNALS, load_split
ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / 'scripts'
TAB_DIR = ROOT / 'results' / 'tables'
REPORT_DIR = ROOT / 'results' / 'reports'
PLOT_DIR = ROOT / 'plots' / 'resampling'
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR):
    d.mkdir(parents = True, exist_ok = True)
META_COLS = [
    'split',
    'subject',
    'class',
    'segment_idx',
    'segment_id']
RAW_N_SAMPLES = 1118
TARGET_N_SAMPLES = 1022

def load_script_module(script_name = None, module_name = None):
    path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
# WARNING: Decompyle incomplete


def resample_linear(x = None, target_len = None):
    if x.size == target_len:
        return x.astype(np.float32, copy = False)
    xp = None.arange(x.size, dtype = np.float64)
    xq = np.linspace(0, x.size - 1, target_len, dtype = np.float64)
    return np.interp(xq, xp, x.astype(np.float64)).astype(np.float32)


def resample_poly_safe(x = None, target_len = None):
    if x.size == target_len:
        return x.astype(np.float32, copy = False)
    y = None(x.astype(np.float64), up = target_len, down = x.size, padtype = 'line')
    if y.size > target_len:
        y = y[:target_len]
    elif y.size < target_len:
        pad_value = float(y[-1]) if y.size else 0
        y = np.pad(y, (0, target_len - y.size), mode = 'constant', constant_values = pad_value)
    return y.astype(np.float32)


def resample_signal(x = None, method = None, target_len = None):
    if method == 'linear':
        return resample_linear(x, target_len)
    if None == 'poly':
        return resample_poly_safe(x, target_len)
    raise None(method)


def extract_tf_features(tf_mod = None, tensor = None, meta = None, split = ('meta', 'pd.DataFrame', 'split', 'str', 'return', 'tuple[pd.DataFrame, dict[str, int]]')):
    rows = []
    fail_counts = { }
# WARNING: Decompyle incomplete


def merge_feature_tables(physio = None, tf = None, raw = None):
    pass
# WARNING: Decompyle incomplete


def build_resampled_tensor(split = None, method = None, target_len = None):
    (tensor_raw, meta) = load_split(split, n_samples = RAW_N_SAMPLES)
    out = np.zeros((len(meta), len(SIGNALS), target_len), dtype = np.float32)
    lengths = np.zeros((len(meta), len(SIGNALS)), dtype = np.int32)
    for i in tqdm(range(len(meta)), desc = f'''resample {split} {method}'''):
        for s_i, sig in enumerate(SIGNALS):
            raw_len = int(meta.iloc[i][f'''raw_len_{sig}'''])
            x = tensor_raw[(i, s_i, :raw_len)]
            lengths[(i, s_i)] = raw_len
            out[(i, s_i)] = resample_signal(x, method = method, target_len = target_len)
    return (out, meta, lengths)


def save_example_plots(split, tensor_raw = None, tensor_map = None, meta = None, lengths = ('split', 'str', 'tensor_raw', 'np.ndarray', 'tensor_map', 'dict[str, np.ndarray]', 'meta', 'pd.DataFrame', 'lengths', 'np.ndarray', 'return', 'list[Path]')):
    out_paths = []
    want_lengths = [
        990,
        1022,
        1118]
    picks = []
    for wl in want_lengths:
        idx = np.where(lengths[(:, SIGNALS.index('Resp'))] == wl)[0]
        if not len(idx):
            continue
        picks.append(int(idx[0]))
    if not picks:
        picks = [
            0,
            min(1, len(meta) - 1),
            min(2, len(meta) - 1)]
    picks = picks[:3]
    for idx in picks:
        subject = int(meta.iloc[idx]['subject'])
        cls = meta.iloc[idx]['class']
        seg = int(meta.iloc[idx]['segment_idx'])
        (fig, axes) = plt.subplots(2, 1, figsize = (10, 6), sharex = False)
        for ax, sig in zip(axes, ('Resp', 'Bvp')):
            s_i = SIGNALS.index(sig)
            raw_len = int(lengths[(idx, s_i)])
            raw = tensor_raw[(idx, s_i, :raw_len)]
            ax.plot(np.arange(raw_len), raw, label = f'''raw_{raw_len}''', linewidth = 1.4, color = 'black')
            trunc = raw[:TARGET_N_SAMPLES] if raw_len >= TARGET_N_SAMPLES else np.pad(raw, (0, TARGET_N_SAMPLES - raw_len), constant_values = np.nan)
            ax.plot(np.arange(TARGET_N_SAMPLES), trunc, label = 'truncate/pad_1022', alpha = 0.7)
            for method, tensor in tensor_map.items():
                ax.plot(np.arange(TARGET_N_SAMPLES), tensor[(idx, s_i)], label = f'''{method}_1022''', alpha = 0.8)
            ax.set_title(f'''{sig} | subj {subject} | {cls} seg {seg} | raw_len={raw_len}''')
            ax.set_ylabel(sig)
            ax.legend(loc = 'best', fontsize = 8)
        axes[-1].set_xlabel('sample')
        fig.tight_layout()
        fp = PLOT_DIR / f'''resample_example_{split}_{subject}_{cls}_{seg}.png'''
        fig.savefig(fp, dpi = 140)
        plt.close(fig)
        out_paths.append(fp)
    return out_paths


def tagged_path(base_name = None, tag = None, directory = None):
    (stem, ext) = base_name.rsplit('.', 1)
    return directory / f'''{stem}_{tag}.{ext}'''


def write_report(rows = None, example_paths = None, elapsed_s = None, methods = ('rows', 'list[dict]', 'example_paths', 'list[Path]', 'elapsed_s', 'float', 'methods', 'list[str]', 'return', 'None')):
    lines = [
        '# 30 — Resampled feature tables\n']
    lines.append(f'''- runtime: {elapsed_s:.1f}s''')
    lines.append(f'''- target length: {TARGET_N_SAMPLES}''')
    lines.append(f'''- source length for cache loading: {RAW_N_SAMPLES}''')
    lines.append(f'''- methods: {', '.join(methods)}''')
    lines.append('')
    lines.append('## Outputs')
    lines.append('')
    lines.append('| method | split | merged shape | tf failures | merged parquet |')
    lines.append('|---|---|---:|---:|---|')
    for row in rows:
        lines.append(f'''| {row['method']} | {row['split']} | {row['merged_shape']} | {row['tf_failures']} | `{row['merged_fp']}` |''')
    lines.append('')
    lines.append('## Example Plots')
    lines.append('')
    for fp in example_paths:
        lines.append(f'''- `{fp}`''')
    (REPORT_DIR / '30_build_resampled_feature_tables.md').write_text('\n'.join(lines))


def main(methods = None, target_len = None):
    t0 = time.time()
    if not methods:
        methods
    methods = [
        'linear',
        'poly']
    raw_mod = load_script_module('02_raw_stats.py', 'ai4pain_raw_stats_resample')
    physio_mod = load_script_module('03_physio_features.py', 'ai4pain_physio_resample')
    tf_mod = load_script_module('04_tfdomain_features.py', 'ai4pain_tf_resample')
    report_rows = []
    example_paths = []
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs = '*', default = [
        'linear',
        'poly'], choices = [
        'linear',
        'poly'])
    parser.add_argument('--target-len', type = int, default = TARGET_N_SAMPLES)
    args = parser.parse_args()
    main(methods = args.methods, target_len = args.target_len)
    return None
