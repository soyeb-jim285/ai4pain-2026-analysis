# Source Generated with Decompyle++
# File: 02_raw_stats.cpython-312.pyc (Python 3.12)

__doc__ = 'Raw-signal descriptive statistics for AI4Pain 2026.\n\nPer-segment summary stats (using np.nan* aware functions) for each of the 4\nsignals, aggregated per class/subject, saved to parquet/CSV and visualised.\n\nRun:\n    uv run python scripts/02_raw_stats.py\nfrom the repo root.\n'
from __future__ import annotations
import sys
import warnings
from pathlib import Path
from matplotlib.pyplot import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as sp_signal
from scipy import stats as sp_stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import CLASSES, SIGNALS, load_split
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_ROOT / 'results' / 'tables'
REPORTS_DIR = ANALYSIS_ROOT / 'results' / 'reports'
PLOTS_DIR = ANALYSIS_ROOT / 'plots' / 'raw_stats'
STAT_NAMES = [
    'mean',
    'std',
    'min',
    'max',
    'median',
    'iqr',
    'skew',
    'kurtosis',
    'range',
    'mad',
    'rms',
    'cv',
    'diff_std',
    'n_extrema']

def _nan_iqr(x = None):
    (q75, q25) = np.nanpercentile(x, [
        75,
        25])
    return float(q75 - q25)


def _nan_mad(x = None):
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def _nan_rms(x = None):
    return float(np.sqrt(np.nanmean(x * x)))


def _n_extrema(x = None):
    '''Count local extrema on the standardised, NaN-stripped signal.'''
    x = x[~np.isnan(x)]
    if x.size < 3:
        return 0
    std = x.std()
    if std < 1e-12:
        return 0
    z = (x - x.mean()) / std
    (pk_pos, _) = sp_signal.find_peaks(z)
    (pk_neg, _) = sp_signal.find_peaks(-z)
    return int(pk_pos.size + pk_neg.size)


def compute_stats(x = None):
    '''All per-segment stats for one 1-D signal with trailing NaNs allowed.'''
    warnings.catch_warnings()
    warnings.simplefilter('ignore', category = RuntimeWarning)
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x))
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    med = float(np.nanmedian(x))
    iqr = _nan_iqr(x)
    sk = float(sp_stats.skew(x, nan_policy = 'omit', bias = False))
    ku = float(sp_stats.kurtosis(x, nan_policy = 'omit', bias = False))
    rng = mx - mn
    mad = _nan_mad(x)
    rms = _nan_rms(x)
    cv = float(std / mean) if abs(mean) > 1e-12 else float('nan')
    valid = x[~np.isnan(x)]
    diff_std = float(np.nanstd(np.diff(valid))) if valid.size > 1 else float('nan')
    n_ext = _n_extrema(x)
    None(None, None)
# WARNING: Decompyle incomplete


def per_segment_stats(tensor = None, meta = None):
    '''Return a wide dataframe with one row per segment and columns
    {signal}_{stat} for every (signal, stat) pair plus metadata.'''
    n_seg = tensor.shape[0]
    rows = []
    for i in range(n_seg):
        row = { }
        for s_i, sig in enumerate(SIGNALS):
            stats_i = compute_stats(tensor[(i, s_i)])
            for k, v in stats_i.items():
                row[f'''{sig}_{k}'''] = v
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    out = pd.concat([
        meta[[
            'split',
            'subject',
            'class',
            'segment_idx',
            'segment_id']].reset_index(drop = True),
        stats_df], axis = 1)
    return out


def _stat_columns():
    '''Return list of (column, signal, stat) for all stat columns.'''
    out = []
    for sig in SIGNALS:
        for st in STAT_NAMES:
            out.append((f'''{sig}_{st}''', sig, st))
    return out


def aggregate_by_class(df = None):
    '''Long-format: rows = (split, class, signal, stat) with mean/std/n.'''
    records = []
    for split, cls in df.groupby([
        'split',
        'class'], sort = False):
        sub = None
        for col, sig, st in _stat_columns():
            vals = sub[col].to_numpy()
            records.append({
                'split': split,
                'class': cls,
                'signal': sig,
                'stat': st,
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals)),
                'n': int(np.isfinite(vals).sum()) })
    return pd.DataFrame(records)


def aggregate_by_subject_class(df = None):
    '''One row per (split, subject, class, signal) with mean of each stat.'''
    records = []
    for split, subj, cls in df.groupby([
        'split',
        'subject',
        'class'], sort = False):
        sub = None
        for sig in SIGNALS:
            row = {
                'split': split,
                'subject': subj,
                'class': cls,
                'signal': sig }
            for st in STAT_NAMES:
                row[st] = float(np.nanmean(sub[f'''{sig}_{st}'''].to_numpy()))
            records.append(row)
    return pd.DataFrame(records)


def anova_quick(df_seg = None):
    '''One-way ANOVA over subject-averaged values (train split) across 3 classes.

    Returns rows {signal, stat, F, p, eta_sq, n_nopain, n_painarm, n_painhand}.
    '''
    pass
# WARNING: Decompyle incomplete


def plot_boxplots(df_seg = None, stat = None, out_fp = None):
    train = df_seg[df_seg['split'] == 'train']
    (fig, axes) = plt.subplots(1, len(SIGNALS), figsize = (4 * len(SIGNALS), 4), sharey = False)
    for ax, sig in zip(axes, SIGNALS):
        col = f'''{sig}_{stat}'''
        sns.boxplot(data = train, x = 'class', y = col, order = list(CLASSES), ax = ax, showfliers = True, palette = 'Set2')
        ax.set_title(f'''{sig} — {stat}''')
        ax.set_xlabel('')
        ax.set_ylabel(col)
    fig.suptitle(f'''Per-segment {stat} by class (train)''', y = 1.02)
    fig.tight_layout()
    fig.savefig(out_fp, dpi = 130, bbox_inches = 'tight')
    plt.close(fig)


def plot_subject_class_heatmap(subj_df = None, signal = None, out_fp = None):
    train = subj_df[(subj_df['split'] == 'train') & (subj_df['signal'] == signal)]
    pivot = train.pivot(index = 'subject', columns = 'class', values = 'mean').reindex(columns = list(CLASSES)).sort_index()
    glob = float(np.nanmean(pivot.to_numpy()))
    span = float(np.nanmax(np.abs(pivot.to_numpy() - glob)))
    if np.isfinite(span) or span == 0:
        span = 1
    (fig, ax) = plt.subplots(figsize = (4.5, max(6, 0.22 * len(pivot))))
    sns.heatmap(pivot, cmap = 'RdBu_r', center = glob, vmin = glob - span, vmax = glob + span, annot = False, cbar_kws = {
        'label': f'''{signal} per-segment mean''' }, ax = ax)
    ax.set_title(f'''{signal}: subject × class mean (train)\nglobal mean={glob:.3f}''')
    ax.set_ylabel('subject')
    ax.set_xlabel('class')
    fig.tight_layout()
    fig.savefig(out_fp, dpi = 130, bbox_inches = 'tight')
    plt.close(fig)


def plot_cv_across_subjects(df_seg = None, signal = None, out_fp = None):
    pass
# WARNING: Decompyle incomplete

# WARNING: Decompyle incomplete
