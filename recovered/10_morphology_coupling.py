# Source Generated with Decompyle++
# File: 10_morphology_coupling.cpython-312.pyc (Python 3.12)

'''PPG-morphology + cross-signal coupling features for AI4Pain 2026.

Purpose
-------
Previous pass (06_class_tests.py) found 0/243 aggregate features surviving
FDR for PainArm vs PainHand. This script probes a different surface:
  - PPG (BVP) waveform morphology (systolic peak, dicrotic notch, rise/fall,
    augmentation index, 2nd-derivative acceleration peaks).
  - BVP envelope / HR-vs-RESP phase coupling (PLV, Kuiper V, RSA proxy).
  - BVP-envelope vs EDA lagged cross-correlation.
  - RESP-EDA coupling (raw + bandpassed r, lagged xcorr).
  - Mutual information + distance correlation across the 6 signal pairs.

Subject-z normalisation is mandatory (applied per subject, per signal).

Outputs
-------
  results/tables/morphology_coupling_features.parquet
  results/tables/morphology_coupling_tests.csv
  results/reports/10_morphology_coupling_summary.md
  plots/armhand_morphology/*.png

Run:
    uv run python scripts/10_morphology_coupling.py
'''
from __future__ import annotations
import sys
import warnings
from itertools import combinations
from pathlib import Path
from matplotlib.pyplot import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as spsig
from scipy import stats as spstats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import SFREQ, SIGNALS, load_split
warnings.filterwarnings('ignore')
np.random.seed(42)

try:
    
    api
    from statsmodels.regression.mixed_linear_model import MixedLM
    import statsmodels.formula.api, formula
    HAVE_LMM = True
    
    try:
        import dcor
        HAVE_DCOR = True
        ANALYSIS_DIR = Path(__file__).resolve().parents[1]
        TABLES_DIR = ANALYSIS_DIR / 'results' / 'tables'
        REPORTS_DIR = ANALYSIS_DIR / 'results' / 'reports'
        PLOTS_DIR = ANALYSIS_DIR / 'plots' / 'armhand_morphology'
        for d in (TABLES_DIR, REPORTS_DIR, PLOTS_DIR):
            d.mkdir(parents = True, exist_ok = True)
        META_COLS = [
            'split',
            'subject',
            'class',
            'segment_idx',
            'segment_id']
        BVP_IDX = SIGNALS.index('Bvp')
        EDA_IDX = SIGNALS.index('Eda')
        RESP_IDX = SIGNALS.index('Resp')
        SPO2_IDX = SIGNALS.index('SpO2')
        
        def _strip_nan(x = None):
            '''Strip trailing NaN, linearly interpolate any interior NaN.'''
            mask = ~np.isnan(x)
            if not mask.any():
                return np.array([], dtype = np.float32)
            last = None(np.flatnonzero(mask)[-1]) + 1
            x = x[:last]
            if np.isnan(x).any():
                idx = np.arange(len(x))
                good = ~np.isnan(x)
                if good.sum() < 2:
                    return np.array([], dtype = np.float32)
                x = None.interp(idx, idx[good], x[good]).astype(np.float32)
            return x.astype(np.float32)

        
        def _butter_filt(x = None, low = None, high = None, sfreq = (SFREQ, 3), order = ('x', 'np.ndarray', 'low', 'float | None', 'high', 'float | None', 'sfreq', 'float', 'order', 'int', 'return', 'np.ndarray')):
            if len(x) < max(3 * order + 1, 15):
                return x.astype(np.float32)
            nyq = None / 2
        # WARNING: Decompyle incomplete

        
        def _zscore(x = None):
            s = np.std(x)
            if not s <= 0 or np.isfinite(s):
                return x - np.mean(x)
            return (None - np.mean(x)) / s

        
        def subject_z_normalise(tensor = None, meta = None):
            '''Per subject, per signal: subtract the subject+signal mean and divide by
    its std, computed across all (segment, sample) pairs for that subject.'''
            out = tensor.copy()
            for sid, idxs in meta.groupby('subject').indices.items():
                sub = out[list(idxs)]
                for s in range(sub.shape[1]):
                    flat = sub[(:, s, :)].reshape(-1)
                    flat = flat[~np.isnan(flat)]
                    if len(flat) < 10:
                        continue
                    mu = float(np.mean(flat))
                    sd = float(np.std(flat))
                    if not sd <= 0 or np.isfinite(sd):
                        sd = 1
                    out[(list(idxs), s, :)] = (out[(list(idxs), s, :)] - mu) / sd
            return out

        
        def ppg_morphology(bvp = None, sfreq = None):
            '''Per-segment PPG waveform morphology features.'''
            pass
        # WARNING: Decompyle incomplete

        
        def _analytic(x = None):
            
            try:
                return spsig.hilbert(x - np.mean(x))
            except Exception:
                return 


        
        def _phase(x = None):
            return np.angle(_analytic(x))

        
        def _envelope(x = None):
            return np.abs(_analytic(x))

        
        def _kuiper_v(phi = None):
            """Kuiper's V statistic on circular data in [-pi, pi]."""
            u = (phi + np.pi) / (2 * np.pi)
            u = np.sort(u)
            n = len(u)
            if n < 2:
                return np.nan
            i = None.arange(1, n + 1)
            D_plus = np.max(i / n - u)
            D_minus = np.max(u - (i - 1) / n)
            return float((D_plus + D_minus) * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)))

        
        def coupling_features(bvp, eda = None, resp = None, spo2 = None, spo2_valid = (SFREQ,), sfreq = ('bvp', 'np.ndarray', 'eda', 'np.ndarray', 'resp', 'np.ndarray', 'spo2', 'np.ndarray | None', 'spo2_valid', 'bool', 'sfreq', 'float', 'return', 'dict[str, float]')):
            feats = { }
            L = min(len(bvp), len(eda), len(resp))
        # WARNING: Decompyle incomplete

        
        def _mutual_information(x = None, y = None, bins = None):
            x = np.asarray(x, dtype = float)
            y = np.asarray(y, dtype = float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if len(x) < 20 and np.std(x) == 0 or np.std(y) == 0:
                return np.nan
            (hist, _, _) = None.histogram2d(x, y, bins = bins)
            p_xy = hist / hist.sum()
            p_x = p_xy.sum(axis = 1, keepdims = True)
            p_y = p_xy.sum(axis = 0, keepdims = True)
            np.errstate(invalid = 'ignore', divide = 'ignore')
            mi = np.nansum(p_xy * (np.log(p_xy + 1e-12) - np.log(p_x + 1e-12) - np.log(p_y + 1e-12)))
            None(None, None)
        # WARNING: Decompyle incomplete

        
        def _distance_correlation(x = None, y = None, subsample = None):
            x = np.asarray(x, dtype = float)
            y = np.asarray(y, dtype = float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if len(x) < 20 and np.std(x) == 0 or np.std(y) == 0:
                return np.nan
            if None:
                
                try:
                    return float(dcor.distance_correlation(x[::subsample], y[::subsample]))
                    xs = x[::subsample]
                    ys = y[::subsample]
                    n = len(xs)
                    if n < 10:
                        return np.nan
                    a = None.abs(xs[(:, None)] - xs[(None, :)])
                    b = np.abs(ys[(:, None)] - ys[(None, :)])
                    a_mean_r = a.mean(axis = 1, keepdims = True)
                    a_mean_c = a.mean(axis = 0, keepdims = True)
                    a_grand = a.mean()
                    b_mean_r = b.mean(axis = 1, keepdims = True)
                    b_mean_c = b.mean(axis = 0, keepdims = True)
                    b_grand = b.mean()
                    A = (a - a_mean_r - a_mean_c) + a_grand
                    B = (b - b_mean_r - b_mean_c) + b_grand
                    dcov2_xy = (A * B).mean()
                    dcov2_xx = (A * A).mean()
                    dcov2_yy = (B * B).mean()
                    if dcov2_xx <= 0 and dcov2_yy <= 0 or dcov2_xy < 0:
                        return 0
                    return float(np.sqrt(dcov2_xy / np.sqrt(dcov2_xx * dcov2_yy)))
                except Exception:
                    continue


        
        def extract_segment_features(seg = None, spo2_valid = None):
            bvp = _strip_nan(seg[BVP_IDX])
            eda = _strip_nan(seg[EDA_IDX])
            resp = _strip_nan(seg[RESP_IDX])
            spo2 = _strip_nan(seg[SPO2_IDX])
            feats = { }
            
            try:
                feats.update(ppg_morphology(bvp))
                
                try:
                    feats.update(coupling_features(bvp, eda, resp, spo2, spo2_valid))
                    return feats
                    except Exception:
                        continue
                except Exception:
                    return feats



        
        def build_feature_matrix():
            const_df = pd.read_csv(TABLES_DIR / 'inventory_constant_segments.csv')
            const_set = set(const_df[const_df['signal'] == 'SpO2']['segment_id'].astype(str).tolist())
            all_rows = []
        # WARNING: Decompyle incomplete

        
        def cliffs_delta(x = None, y = None):
            x = np.asarray(x, dtype = float)
            y = np.asarray(y, dtype = float)
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            ny = len(y)
            nx = len(x)
            if nx == 0 or ny == 0:
                return np.nan
            gt = None(np.sum(x[(:, None)] > y[(None, :)]))
            lt = int(np.sum(x[(:, None)] < y[(None, :)]))
            return float(gt - lt) / (nx * ny)

        
        def paired_subject_means(df = None, feature = None):
            arm = df[df['class'] == 'PainArm'].groupby('subject', as_index = False)[feature].mean()
            hand = df[df['class'] == 'PainHand'].groupby('subject', as_index = False)[feature].mean()
            m = arm.merge(hand, on = 'subject', suffixes = ('_arm', '_hand')).dropna()
            return (m[f'''{feature}_arm'''].to_numpy(dtype = float), m[f'''{feature}_hand'''].to_numpy(dtype = float), m['subject'].to_numpy())

        
        def fit_lmm(df_tr = None, feature = None):
            if not HAVE_LMM:
                return np.nan
            sub = None[df_tr['class'].isin([
                'PainArm',
                'PainHand'])][[
                'subject',
                'class',
                feature]].dropna()
            if sub['subject'].nunique() < 5 or sub[feature].std() == 0:
                return np.nan
            sub = None.copy()
            sub['is_hand'] = (sub['class'] == 'PainHand').astype(int)
            
            try:
                md = MixedLM.from_formula(f'''{feature} ~ is_hand''', groups = 'subject', data = sub)
                mdf = md.fit(method = 'lbfgs', reml = True, disp = False)
                p = float(mdf.pvalues.get('is_hand', np.nan))
                return p
            except Exception:
                return 


        
        def run_armhand_tests(df = None, features = None):
            tr = df[df['split'] == 'train'].copy()
            rows = []
            for feat in tqdm(features, desc = 'armhand tests', ncols = 88):
                (x, y, _) = paired_subject_means(tr, feat)
                row = {
                    'feature': feat,
                    'n_subjects': int(len(x)) }
                row['cliffs_delta'] = cliffs_delta(x, y)
                d = x - y
                d = d[np.isfinite(d) & (d != 0)]
                row['mean_arm'] = float(np.mean(x)) if len(x) else np.nan
                row['mean_hand'] = float(np.mean(y)) if len(y) else np.nan
                row['direction'] = 'Arm > Hand' if row['mean_arm'] > row['mean_hand'] else 'Arm < Hand'
                row['lmm_p'] = fit_lmm(tr, feat)
                rows.append(row)
            out = pd.DataFrame(rows)
            
            def _fam(f = None):
                if f.startswith('ppg_'):
                    return 'morphology'
                if f.startswith('mi_') or f.startswith('dcor_'):
                    return 'coupling_nonlinear'
                return 'coupling_linear'

            out['family'] = out['feature'].apply(_fam)
            for test_col in ('wilcoxon_p', 'lmm_p'):
                adj_col = test_col.replace('_p', '_p_fdr')
                out[adj_col] = np.nan
                for fam, sub in out.groupby('family'):
                    p = sub[test_col].to_numpy()
                    mask = np.isfinite(p)
                    if mask.sum() == 0:
                        continue
                    (_, p_adj, _, _) = multipletests(p[mask], alpha = 0.05, method = 'fdr_bh')
                    adj = np.full_like(p, np.nan, dtype = float)
                    adj[mask] = p_adj
                    out.loc[(sub.index, adj_col)] = adj
            p = out['wilcoxon_p'].to_numpy()
            mask = np.isfinite(p)
            adj_all = np.full_like(p, np.nan, dtype = float)
            if mask.sum() > 0:
                (_, p_adj, _, _) = multipletests(p[mask], alpha = 0.05, method = 'fdr_bh')
                adj_all[mask] = p_adj
            out['wilcoxon_p_fdr_global'] = adj_all
            return out
            except Exception:
                row['wilcoxon_W'] = np.nan
                row['wilcoxon_p'] = np.nan
                continue

        
        def validation_reproducibility(df = None, top_feats = None):
            val = df[df['split'] == 'validation']
            tr = df[df['split'] == 'train']
            rows = []
            for feat in top_feats:
                (tr_arm, tr_hand, _) = paired_subject_means(tr, feat)
                (va_arm, va_hand, _) = paired_subject_means(val, feat)
                tr_eff = float(np.mean(tr_arm) - np.mean(tr_hand)) if len(tr_arm) else np.nan
                va_eff = float(np.mean(va_arm) - np.mean(va_hand)) if len(va_arm) else np.nan
                if np.sign(tr_eff) == np.sign(va_eff):
                    np.sign(tr_eff) == np.sign(va_eff)
                    if np.isfinite(tr_eff):
                        np.isfinite(tr_eff)
                preserved = np.isfinite(va_eff)
                rows.append({
                    'feature': feat,
                    'train_arm_minus_hand': tr_eff,
                    'val_arm_minus_hand': va_eff,
                    'train_direction_preserved_in_val': bool(preserved),
                    'val_n_subjects': int(len(va_arm)) })
            return pd.DataFrame(rows)

        
        def plot_ppg_beat_template(df_feat = None, tensor_tr = None, meta_tr = None):
            '''Per-class average beat template aligned on systolic peak.'''
            (fig, ax) = plt.subplots(figsize = (8, 5))
            subjects = sorted(meta_tr['subject'].unique())
            win = int(0.5 * SFREQ)
            for cls, color in (('PainArm', '#dd8452'), ('PainHand', '#c44e52')):
                subj_templates = []
                for sid in subjects:
                    mask = (meta_tr['subject'] == sid) & (meta_tr['class'] == cls)
                    idxs = meta_tr.index[mask].to_numpy()
                    beats = []
                    for i in idxs:
                        bvp = _strip_nan(tensor_tr[(i, BVP_IDX)])
                        if len(bvp) < 200:
                            continue
                        x = _butter_filt(bvp, 0.5, 8, SFREQ)
                        x = _zscore(x)
                        prom = 0.3 * float(np.std(x)) if np.std(x) > 0 else None
                        (peaks, _) = spsig.find_peaks(x, distance = int(0.4 * SFREQ), prominence = prom)
                        for pk in peaks:
                            lo = pk - win
                            hi = pk + win
                            if lo < 0 or hi > len(x):
                                continue
                            beats.append(x[lo:hi])
                    if not beats:
                        continue
                    subj_templates.append(np.mean(np.stack(beats, axis = 0), axis = 0))
                if not subj_templates:
                    continue
                arr = np.stack(subj_templates, axis = 0)
                mean = arr.mean(axis = 0)
                sem = arr.std(axis = 0, ddof = 1) / np.sqrt(arr.shape[0])
                t = np.arange(-win, win) / SFREQ
                ax.plot(t, mean, color = color, linewidth = 2, label = f'''{cls} (n={arr.shape[0]} subj)''')
                ax.fill_between(t, mean - sem, mean + sem, color = color, alpha = 0.25)
            ax.axvline(0, color = 'k', linewidth = 0.5, linestyle = '--')
            ax.set_xlabel('Time relative to systolic peak (s)')
            ax.set_ylabel('Subject-z BVP')
            ax.set_title('Mean PPG beat template: PainArm vs PainHand (train)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / 'ppg_beat_template_per_class.png')
            plt.close(fig)

        
        def plot_resp_bvp_phase_hist(df_feat = None):
            tr = df_feat[df_feat['split'] == 'train']
            arm = tr[tr['class'] == 'PainArm']['resp_bvp_mean_phase_diff'].dropna().to_numpy()
            hand = tr[tr['class'] == 'PainHand']['resp_bvp_mean_phase_diff'].dropna().to_numpy()
            (ax1, ax2) = (fig,)
            for ax, vals, cls, color in ((ax1, arm, 'PainArm', '#dd8452'), (ax2, hand, 'PainHand', '#c44e52')):
                bins = np.linspace(-(np.pi), np.pi, 25)
                (counts, edges) = np.histogram(vals, bins = bins)
                widths = np.diff(edges)
                centers = (edges[:-1] + edges[1:]) / 2
                ax.bar(centers, counts, width = widths, color = color, alpha = 0.7, edgecolor = 'k')
                ax.set_title(f'''{cls} (n={len(vals)} segments)''')
            fig.suptitle('RESP-BVP mean phase difference distribution')
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / 'resp_bvp_phase_hist.png')
            plt.close(fig)

        
        def plot_lagged_xcorr_bvp_eda(tensor_tr = None, meta_tr = None):
            subjects = sorted(meta_tr['subject'].unique())
            max_lag = int(5 * SFREQ)
            step = max(1, int(0.05 * SFREQ))
            lags = np.arange(-max_lag, max_lag + 1, step)
            (fig, ax) = plt.subplots(figsize = (8, 5))
            for cls, color in (('PainArm', '#dd8452'), ('PainHand', '#c44e52')):
                subj_curves = []
                for sid in subjects:
                    mask = (meta_tr['subject'] == sid) & (meta_tr['class'] == cls)
                    idxs = meta_tr.index[mask].to_numpy()
                    seg_curves = []
                    for i in idxs:
                        bvp = _strip_nan(tensor_tr[(i, BVP_IDX)])
                        eda = _strip_nan(tensor_tr[(i, EDA_IDX)])
                        L = min(len(bvp), len(eda))
                        if L < int(3 * SFREQ):
                            continue
                        bvp = bvp[:L]
                        eda = eda[:L]
                        env = _envelope(_butter_filt(bvp, 0.5, 4, SFREQ))
                        eda_lp = _butter_filt(eda, None, 2, SFREQ)
                        a0 = _zscore(env)
                        b0 = _zscore(eda_lp)
                        corrs = np.zeros(len(lags), dtype = float)
                        for k, lag in enumerate(lags):
                            if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
                                corrs[k] = float(np.corrcoef(a, b)[(0, 1)])
                                continue
                            corrs[k] = np.nan
                        seg_curves.append(corrs)
                    if not seg_curves:
                        continue
                    subj_curves.append(np.nanmean(np.stack(seg_curves, axis = 0), axis = 0))
                if not subj_curves:
                    continue
                arr = np.stack(subj_curves, axis = 0)
                mean = np.nanmean(arr, axis = 0)
                sem = np.nanstd(arr, axis = 0, ddof = 1) / np.sqrt(arr.shape[0])
                t = lags / SFREQ
                ax.plot(t, mean, color = color, linewidth = 2, label = f'''{cls} (n={arr.shape[0]} subj)''')
                ax.fill_between(t, mean - sem, mean + sem, color = color, alpha = 0.25)
            ax.axhline(0, color = 'k', linewidth = 0.5)
            ax.axvline(0, color = 'k', linewidth = 0.5, linestyle = '--')
            ax.set_xlabel('Lag (s) — positive = BVP-env leads EDA')
            ax.set_ylabel('Pearson r')
            ax.set_title('BVP envelope vs EDA cross-correlation (train, subject-averaged)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / 'lagged_xcorr_bvp_eda.png')
            plt.close(fig)
            return None
            except Exception:
                continue

        
        def plot_sync_index_violin(df_feat = None):
            tr = df_feat[df_feat['split'] == 'train']
            sub = tr[tr['class'].isin([
                'PainArm',
                'PainHand'])]
            if 'resp_bvp_plv' not in sub.columns:
                return None
            (fig, ax) = plt.subplots(figsize = (6, 5))
            sns.violinplot(data = sub, x = 'class', y = 'resp_bvp_plv', palette = {
                'PainArm': '#dd8452',
                'PainHand': '#c44e52' }, ax = ax, inner = 'quartile')
            ax.set_title('RESP-BVP phase-locking value (PLV)')
            ax.set_ylabel('PLV')
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / 'synchronization_index_violin.png')
            plt.close(fig)

        
        def plot_mi_heatmap(df_feat = None):
            tr = df_feat[df_feat['split'] == 'train']
            signals4 = [
                'bvp',
                'eda',
                'resp',
                'spo2']
            (fig, axes) = plt.subplots(1, 2, figsize = (11, 5))
            for ax, cls in zip(axes, [
                'PainArm',
                'PainHand']):
                mat = np.full((4, 4), np.nan)
                sub = tr[tr['class'] == cls]
                for i, a in enumerate(signals4):
                    for j, b in enumerate(signals4):
                        if i == j:
                            mat[(i, j)] = np.nan
                            continue
                        key = f'''mi_{a}_{b}''' if f'''mi_{a}_{b}''' in tr.columns else f'''mi_{b}_{a}'''
                        if not key in sub.columns:
                            continue
                        mat[(i, j)] = sub[key].mean()
                sns.heatmap(mat, annot = True, fmt = '.3f', xticklabels = signals4, yticklabels = signals4, cmap = 'viridis', ax = ax, cbar_kws = {
                    'label': 'MI (nats)' })
                ax.set_title(f'''{cls} — class-mean MI''')
            fig.suptitle('Mutual information between signal pairs (train subject-means)')
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / 'mi_heatmap.png')
            plt.close(fig)

        
        def plot_top20_wilcoxon_bar(tests = None):
            t = tests.dropna(subset = [
                'wilcoxon_W']).copy()
            t['abs_W'] = t['wilcoxon_W'].abs()
            t = t.sort_values('abs_W', ascending = False).head(20)
            (fig, ax) = plt.subplots(figsize = (10, 8))
        # WARNING: Decompyle incomplete

        
        def write_report(tests = None, val_df = None, n_features = None, n_train_subj = ('tests', 'pd.DataFrame', 'val_df', 'pd.DataFrame', 'n_features', 'int', 'n_train_subj', 'int', 'return', 'None')):
            fdr_sig = tests[tests['wilcoxon_p_fdr'] < 0.05]
            fdr_sig_global = tests[tests['wilcoxon_p_fdr_global'] < 0.05]
            top15 = tests.sort_values('wilcoxon_p').head(15).copy()
            fam_sum = tests.groupby('family').agg(n = ('feature', 'count'), n_sig_fdr = ('wilcoxon_p_fdr', (lambda s: int((s < 0.05).sum()))), min_p = ('wilcoxon_p', 'min'), median_abs_delta = ('cliffs_delta', (lambda s: float(np.nanmedian(np.abs(s)))))).reset_index()
            val_lookup = val_df.set_index('feature') if len(val_df) else None
            lines = []
            lines.append('# 10 Morphology + coupling features (PainArm vs PainHand)')
            lines.append('')
            lines.append(f'''- Total features: **{n_features}** (morphology + linear coupling + nonlinear coupling)''')
            lines.append(f'''- Train subjects (paired Wilcoxon): {n_train_subj}''')
            lines.append(f'''- FDR<0.05 within family: **{len(fdr_sig)}**''')
            lines.append(f'''- FDR<0.05 globally (all features, single family): **{len(fdr_sig_global)}**''')
            lines.append('')
            lines.append('## Family breakdown')
            lines.append('')
            lines.append('| family | n features | n FDR<0.05 | min p | median |delta| |')
            lines.append('|--------|-----------:|-----------:|------:|---------------:|')
            for _, r in fam_sum.iterrows():
                lines.append(f'''| {r['family']} | {int(r['n'])} | {int(r['n_sig_fdr'])} | {r['min_p']:.2e} | {r['median_abs_delta']:.3f} |''')
            lines.append('')
            lines.append('## Top 15 features by raw Wilcoxon p')
            lines.append('')
            lines.append("| rank | feature | family | W | p | p_fdr (family) | Cliff's d | sign | dir | val-dir-pres |")
            lines.append('|-----:|---------|--------|---:|---:|---------------:|---------:|------:|-----|:------------:|')
        # WARNING: Decompyle incomplete

        
        def main():
            print('[1/6] Building feature matrix...')
            feats_df = build_feature_matrix()
        # WARNING: Decompyle incomplete

        if __name__ == '__main__':
            main()
            return None
        return None
        except Exception:
            HAVE_LMM = False
            continue
    except Exception:
        HAVE_DCOR = False
        continue


