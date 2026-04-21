# Source Generated with Decompyle++
# File: 23_tierC_tvsymp_eda.cpython-312.pyc (Python 3.12)

'''Tier-C (Paper 3 idea): TVSymp latency on EDA for ARM vs HAND.

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
'''
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
from src.data_loader import SFREQ, SIGNALS, load_split
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / 'results' / 'tables'
REP = ROOT / 'results' / 'reports'
TAB.mkdir(parents = True, exist_ok = True)
REP.mkdir(parents = True, exist_ok = True)
EDA_IDX = SIGNALS.index('Eda')

def _strip_nan(x = None):
    mask = ~np.isnan(x)
    if not mask.any():
        return np.array([], dtype = np.float32)
    last = None(np.flatnonzero(mask)[-1]) + 1
    return x[:last].astype(np.float32)


def _butter(x = None, lo = None, hi = None, fs = (4,), order = ('x', 'np.ndarray', 'lo', 'float | None', 'hi', 'float | None', 'fs', 'float', 'order', 'int', 'return', 'np.ndarray')):
    nyq = fs / 2
# WARNING: Decompyle incomplete


def _phasic_eda(eda = None, fs = None):
    '''Low-pass at 5 Hz, subtract tonic (0.05 Hz low-pass). Paper 3 recipe.'''
    if len(eda) < int(2 * fs):
        return eda
    y = None(eda, None, 5, fs, order = 4)
    tonic = _butter(y, None, 0.05, fs, order = 2)
    phasic = y - tonic
    phasic = phasic - np.mean(phasic)
    return phasic


def _tvsymp(phasic = None, fs = None):
    '''Narrowband 0.08-0.24 Hz + Hilbert envelope, normalised by its std.'''
    if len(phasic) < int(4 * fs):
        return np.full_like(phasic, np.nan, dtype = float)
    band = None(phasic, 0.08, 0.24, fs, order = 4)
    env = np.abs(spsig.hilbert(band))
    sd = float(np.std(env))
    if sd <= 1e-09:
        return np.zeros_like(env)
    return None / sd

FEATURE_NAMES = [
    'tvsymp_peak_amp',
    'tvsymp_peak_time_s',
    'tvsymp_time_to_half_peak_s',
    'tvsymp_rise_slope',
    'tvsymp_auc',
    'tvsymp_centroid_s',
    'tvsymp_mean',
    'tvsymp_std',
    'tvsymp_early_mean_0_3s',
    'tvsymp_late_mean_7_10s',
    'tvsymp_early_minus_late',
    'tvsymp_first_crossing_mean_s']

def extract_tvsymp(eda_raw = None, fs = None):
    pass
# WARNING: Decompyle incomplete


def build_features():
    rows = []
# WARNING: Decompyle incomplete


def paired_wilcoxon_armhand(df = None):
    tr = df[df['split'] == 'train']
    rows = []
    for feat in FEATURE_NAMES:
        arm = tr[tr['class'] == 'PainArm'].groupby('subject')[feat].mean()
        hand = tr[tr['class'] == 'PainHand'].groupby('subject')[feat].mean()
        m = pd.concat([
            arm,
            hand], axis = 1, keys = [
            'arm',
            'hand']).dropna()
        if len(m) < 5:
            rows.append({
                'feature': feat,
                'n': len(m),
                'p': np.nan,
                'mean_arm': np.nan,
                'mean_hand': np.nan })
            continue
        (_, p) = spstats.wilcoxon(m['arm'], m['hand'], zero_method = 'wilcox')
        rows.append({
            'feature': feat,
            'n': len(m),
            'p': float(p),
            'mean_arm': float(m['arm'].mean()),
            'mean_hand': float(m['hand'].mean()) })
    out = pd.DataFrame(rows)
    mask = out['p'].notna()
    p_fdr = np.full(len(out), np.nan)
    if mask.any():
        (_, padj, _, _) = multipletests(out.loc[(mask, 'p')], alpha = 0.05, method = 'fdr_bh')
        p_fdr[mask.to_numpy()] = padj
    out['p_fdr'] = p_fdr
    return out.sort_values('p')
    except Exception:
        p = np.nan
        continue


def _subject_z(X = None, subj = None):
    out = X.astype(float).copy()
    for s in np.unique(subj):
        m = subj == s
        mu = np.nanmean(out[m], axis = 0)
        sd = np.nanstd(out[m], axis = 0)
        sd[sd < 1e-09] = 1
        out[m] = (out[m] - mu) / sd
    return out


def loso_armhand(df = None):
    tr = df[(df['split'] == 'train') & df['class'].isin([
        'PainArm',
        'PainHand'])].copy()
    X = tr[FEATURE_NAMES].to_numpy(dtype = float)
    y = (tr['class'] == 'PainHand').astype(int).to_numpy()
    subj = tr['subject'].to_numpy()
    X = _subject_z(X, subj)
    X = SimpleImputer(strategy = 'median').fit_transform(X)
    X = StandardScaler().fit_transform(X)
    logo = LeaveOneGroupOut()
    scores = []
    for tr_i, te_i in logo.split(X, y, groups = subj):
        clf = LogisticRegression(max_iter = 2000, C = 1, class_weight = 'balanced')
        clf.fit(X[tr_i], y[tr_i])
        scores.append(f1_score(y[te_i], clf.predict(X[te_i]), average = 'macro'))
    scores = np.array(scores)
    return (float(scores.mean()), float(scores.std()), scores)


def val_armhand(df = None):
    tr = df[(df['split'] == 'train') & df['class'].isin([
        'PainArm',
        'PainHand'])]
    va = df[(df['split'] == 'validation') & df['class'].isin([
        'PainArm',
        'PainHand'])]
    if len(va) == 0:
        return float('nan')
    Xt = None(tr[FEATURE_NAMES].to_numpy(float), tr['subject'].to_numpy())
    Xv = _subject_z(va[FEATURE_NAMES].to_numpy(float), va['subject'].to_numpy())
    imp = SimpleImputer(strategy = 'median').fit(Xt)
    sca = StandardScaler().fit(imp.transform(Xt))
    Xt = sca.transform(imp.transform(Xt))
    Xv = sca.transform(imp.transform(Xv))
    yt = (tr['class'] == 'PainHand').astype(int).to_numpy()
    yv = (va['class'] == 'PainHand').astype(int).to_numpy()
    clf = LogisticRegression(max_iter = 2000, C = 1, class_weight = 'balanced')
    clf.fit(Xt, yt)
    return float(f1_score(yv, clf.predict(Xv), average = 'macro'))


def main():
    print('[23] extracting TVSymp features...')
    df = build_features()
    df.to_parquet(TAB / 'tierC_tvsymp_eda_features.parquet', index = False)
    print(f'''[23] wrote {TAB / 'tierC_tvsymp_eda_features.parquet'}  ({len(df)} rows, {len(FEATURE_NAMES)} features)''')
    print('[23] paired Wilcoxon (ARM vs HAND, subject means)...')
    tests = paired_wilcoxon_armhand(df)
    tests.to_csv(TAB / 'tierC_tvsymp_eda_tests.csv', index = False)
    n_sig = int((tests['p_fdr'] < 0.05).sum())
    print('[23] LOSO ARM vs HAND (LogReg + subject-z)...')
    (f1_mean, f1_std, _) = loso_armhand(df)
    f1_val = val_armhand(df)
    fp = open(REP / 'tierC_tvsymp_eda_summary.md', 'w')
    fp.write('# Tier-C TVSymp on EDA — ARM vs HAND\n\n')
    fp.write(f'''- features: {len(FEATURE_NAMES)}\n''')
    fp.write(f'''- segments: {len(df)}\n''')
    fp.write(f'''- FDR<0.05 survivors: **{n_sig} / {len(tests)}**\n''')
    fp.write(f'''- LOSO macro-F1: **{f1_mean:.3f} ± {f1_std:.3f}** (chance = 0.50)\n''')
    fp.write(f'''- Validation macro-F1: **{f1_val:.3f}**\n\n''')
    fp.write('Note: narrowband 0.08-0.24 Hz Butterworth approximates VFCDM used in paper 3. Less selective; feasibility-check only.\n\n')
    fp.write('## All features by p-value\n\n')
    fp.write(tests.to_markdown(index = False))
    fp.write('\n')
    None(None, None)
    print(f'''[23] wrote {REP / 'tierC_tvsymp_eda_summary.md'}''')
    print(f'''[23] LOSO macro-F1 = {f1_mean:.3f} ± {f1_std:.3f} | val = {f1_val:.3f} | FDR<0.05: {n_sig}''')
    return None
    with None:
        if not None:
            pass
    continue

if __name__ == '__main__':
    main()
    return None
