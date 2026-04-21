# Source Generated with Decompyle++
# File: 18_armhand_search.cpython-312.pyc (Python 3.12)

"""Resumable ML hyperparameter + feature-set + preprocessing sweep
for the AI4Pain 2026 ARM-vs-HAND localisation problem.

Targets: best macro-F1 with classical ML (no deep learning, no CNN).
Building on Tier A findings:
  - RESP carries the signal (17/53 features FDR-significant)
  - Top features ranked by |Cliff's delta| are mostly RESP

Search space:
  feature_set  in {all, top80_cliff, resp_only, resp_top20, top40_anova_f}
  preproc      in {subject_z+std, subject_z+robust, subject_z+pca20}
  model        in {LR L2, LR L1, LR EN, RF, ExtraTrees, HistGB, XGB, SVM linear, SVM RBF, kNN, GaussianNB}
  hyperparams  per model, small grid

Plus a stacked ensemble built from top-N base learners after all configs
finish.

Persistence
-----------
- results/tables/armhand_search_perfold.csv     (append-only, one row per (config, fold))
- results/tables/armhand_search_progress.json   ({config_id: status})
- results/tables/armhand_search_summary.csv     (rewritten after each config)
- results/tables/armhand_search_validation.csv  (per-config validation metrics)
- results/tables/armhand_search_top.csv         (final top-K configs)
- results/reports/18_armhand_search_summary.md
- plots/armhand_search/

Resume
------
Re-running the script reads the perfold CSV, identifies completed
(config_id, fold) pairs, and skips them. Configs with all 41 folds done
also skip the LOSO step and only re-run validation if missing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg', force = True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
    warnings.filterwarnings('ignore')
    SEED = 42
    ANALYSIS = Path(__file__).resolve().parents[1]
    TAB_DIR = ANALYSIS / 'results' / 'tables'
    REPORT_DIR = ANALYSIS / 'results' / 'reports'
    PLOT_DIR = ANALYSIS / 'plots' / 'armhand_search'
    for d in (TAB_DIR, REPORT_DIR, PLOT_DIR):
        d.mkdir(parents = True, exist_ok = True)
    PERFOLD_FP = TAB_DIR / 'armhand_search_perfold.csv'
    SUMMARY_FP = TAB_DIR / 'armhand_search_summary.csv'
    VALIDATION_FP = TAB_DIR / 'armhand_search_validation.csv'
    TOP_FP = TAB_DIR / 'armhand_search_top.csv'
    REPORT_FP = REPORT_DIR / '18_armhand_search_summary.md'
    META_COLS = [
        'split',
        'subject',
        'class',
        'segment_idx',
        'segment_id']
    ARM_HAND = ('PainArm', 'PainHand')
    LABEL_MAP = {
        'PainArm': 0,
        'PainHand': 1 }
    
    def load_features():
        fp = TAB_DIR / 'all_features_merged.parquet'
        df = pd.read_parquet(fp)
        tierb_fp = TAB_DIR / 'tierB_derivative_features.parquet'
    # WARNING: Decompyle incomplete

    
    def apply_subject_z(df = None, feat_cols = None):
        out = df.copy()
        means = out.groupby('subject')[feat_cols].transform('mean')
        stds = out.groupby('subject')[feat_cols].transform('std', ddof = 0)
        stds = stds.where(stds > 0, 1)
        out[feat_cols] = ((out[feat_cols] - means) / stds).fillna(0).astype(np.float32)
        return out

    
    def channel_of(name = None):
        n = name.lower()
        hits = []
        for tag in ('bvp', 'eda', 'resp', 'spo2', 'spo_2'):
            if not tag in n:
                continue
            hits.append('spo2' if tag == 'spo_2' else tag)
        hits = sorted(set(hits))
        if not hits:
            return 'other'
        if len(hits) == 1:
            return hits[0]

    
    def feature_strategies(df_pain = None, feat_cols = None):
        """Return mapping name -> ordered list of features.

    Computes Cliff's delta and ANOVA F on TRAIN only; uses outputs from
    Tier A #5 if available."""
        train = df_pain[df_pain['split'] == 'train'].reset_index(drop = True)
        arm_means = train[train['class'] == 'PainArm'].groupby('subject')[feat_cols].mean()
        hand_means = train[train['class'] == 'PainHand'].groupby('subject')[feat_cols].mean()
        common = arm_means.index.intersection(hand_means.index)
        arm_means = arm_means.loc[common]
        hand_means = hand_means.loc[common]
        cliff = { }
        for f in feat_cols:
            a = arm_means[f].to_numpy()
            h = hand_means[f].to_numpy()
            v = ~(np.isnan(a) | np.isnan(h))
            if v.sum() < 5:
                cliff[f] = 0
                continue
            h = h[v]
            a = a[v]
            diffs = a[(:, None)] - h[(None, :)]
            cliff[f] = float((np.sum(diffs > 0) - np.sum(diffs < 0)) / (a.size * h.size))
        cliff_series = pd.Series(cliff).abs().sort_values(ascending = False)
        F_vals = { }
        for f in feat_cols:
            a = arm_means[f].dropna().to_numpy()
            h = hand_means[f].dropna().to_numpy()
            if a.size < 5 or h.size < 5:
                F_vals[f] = 0
                continue
            (F, _) = f_classif(np.concatenate([
                a,
                h]).reshape(-1, 1), np.array([
                0] * len(a) + [
                1] * len(h)))
            F_vals[f] = float(F[0])
        f_series = pd.Series(F_vals).sort_values(ascending = False)
    # WARNING: Decompyle incomplete

    
    def make_preproc(name = None):
        if name == 'std':
            return StandardScaler()
        if None == 'robust':
            return RobustScaler()
        if None == 'pca20':
            return ('compose', StandardScaler(), PCA(n_components = 20, random_state = SEED))
        raise None(name)

    
    def transform(preproc = None, X_train = None, X_test = None):
        if isinstance(preproc, tuple):
            (_, sc, pca) = preproc
            sc.fit(X_train)
            X_train = sc.transform(X_train)
            X_test = sc.transform(X_test)
            n_comp = min(pca.n_components, X_train.shape[1])
            if n_comp != pca.n_components:
                pca = PCA(n_components = n_comp, random_state = SEED)
            else:
                pca = PCA(n_components = pca.n_components, random_state = SEED)
            pca.fit(X_train)
            return (pca.transform(X_train), pca.transform(X_test))
        sc = None.__class__()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        return (X_train, X_test)

    
    def model_factory(model_name = None, params = None):
        p = dict(params)
    # WARNING: Decompyle incomplete

    
    def all_configs():
        '''Focused, time-bounded search.

    Design: instead of a full grid (which would explode XGB and RF), we
    keep small but representative grids per model family and only fan
    out across feature-set / preproc for the cheapest models.
    '''
        cfgs = []
        feature_sets = [
            'all',
            'top40_cliff',
            'top80_cliff',
            'top40_anova',
            'resp_only',
            'resp_top20']
        cheap_preprocs = [
            'std',
            'robust']
    # WARNING: Decompyle incomplete

    PERFOLD_HEADER = [
        'config_id',
        'model',
        'params_str',
        'feature_set',
        'preproc',
        'fold',
        'held_out_subject',
        'n_test',
        'acc',
        'macro_f1',
        'balanced_acc',
        'elapsed_s']
    
    def load_completed_folds():
        if not PERFOLD_FP.exists():
            return { }
        
        try:
            df = pd.read_csv(PERFOLD_FP)
            out = { }
            for cid, sub in df.groupby('config_id'):
                out[cid] = (lambda .0: pass# WARNING: Decompyle incomplete
)(sub['fold'].unique()())
            return out
        except Exception:
            return 


    
    def append_perfold(rows = None):
        if not rows:
            return None
        df = pd.DataFrame(rows)[PERFOLD_HEADER]
        write_header = not PERFOLD_FP.exists()
        df.to_csv(PERFOLD_FP, mode = 'a', header = write_header, index = False)

    
    def write_summary(perfold = None, val_df = None):
        if perfold.empty:
            return pd.DataFrame()
        g = None.groupby([
            'config_id',
            'model',
            'params_str',
            'feature_set',
            'preproc'])
        summ = g['macro_f1'].agg([
            'mean',
            'std',
            'count']).rename(columns = {
            'mean': 'loso_macro_f1_mean',
            'std': 'loso_macro_f1_std',
            'count': 'n_folds' })
        summ['loso_acc_mean'] = g['acc'].mean()
        summ['loso_balanced_acc_mean'] = g['balanced_acc'].mean()
        summ['elapsed_s_total'] = g['elapsed_s'].sum()
        summ = summ.reset_index()
    # WARNING: Decompyle incomplete

    
    def load_validation_results():
        if not VALIDATION_FP.exists():
            return pd.DataFrame(columns = [
                'config_id',
                'val_macro_f1',
                'val_acc',
                'val_balanced_acc'])
        return None.read_csv(VALIDATION_FP)

    
    def append_validation(row = None):
        df = load_validation_results()
        df = df[df['config_id'] != row['config_id']]
        df = pd.concat([
            df,
            pd.DataFrame([
                row])], ignore_index = True)
        df.to_csv(VALIDATION_FP, index = False)

    
    def run_config(cfg = None, df_z = None, fs_map = None, completed_folds = ('cfg', 'dict', 'df_z', 'pd.DataFrame', 'fs_map', 'dict[str, list[str]]', 'completed_folds', 'set[int]', 'return', 'tuple[list[dict], dict | None]')):
        feat_cols = fs_map[cfg['feature_set']]
        train = df_z[df_z['split'] == 'train'].reset_index(drop = True)
        val = df_z[df_z['split'] == 'validation'].reset_index(drop = True)
        train_ah = train[train['class'].isin(ARM_HAND)].reset_index(drop = True)
        val_ah = val[val['class'].isin(ARM_HAND)].reset_index(drop = True)
        X = train_ah[feat_cols].to_numpy()
        y = train_ah['class'].map(LABEL_MAP).to_numpy()
        subjects = train_ah['subject'].to_numpy()
        logo = LeaveOneGroupOut()
        folds = list(logo.split(X, y, groups = subjects))
        new_rows = []
        for tr, te in enumerate(folds):
            if i in completed_folds:
                continue
            t0 = time.time()
            preproc = make_preproc(cfg['preproc'])
            (X_tr, X_te) = transform(preproc, X[tr], X[te])
            mdl = model_factory(cfg['model'], cfg['params'])
            mdl.fit(X_tr, y[tr])
            yhat = mdl.predict(X_te)
            new_rows.append({
                'config_id': cfg['config_id'],
                'model': cfg['model'],
                'params_str': cfg['params_str'],
                'feature_set': cfg['feature_set'],
                'preproc': cfg['preproc'],
                'fold': i,
                'held_out_subject': int(subjects[te][0]),
                'n_test': int(len(te)),
                'acc': accuracy_score(y[te], yhat),
                'macro_f1': f1_score(y[te], yhat, average = 'macro', zero_division = 0),
                'balanced_acc': balanced_accuracy_score(y[te], yhat),
                'elapsed_s': time.time() - t0 })
            append_perfold([
                new_rows[-1]])
        val_row = None
        val_existing = load_validation_results()
        has_val = cfg['config_id'] in val_existing['config_id'].astype(str).values
        if not has_val and val_ah.empty:
            
            try:
                X_va = val_ah[feat_cols].to_numpy()
                y_va = val_ah['class'].map(LABEL_MAP).to_numpy()
                preproc = make_preproc(cfg['preproc'])
                (X_tr_full, X_va_t) = transform(preproc, X, X_va)
                mdl = model_factory(cfg['model'], cfg['params'])
                mdl.fit(X_tr_full, y)
                yhat = mdl.predict(X_va_t)
                val_row = {
                    'config_id': cfg['config_id'],
                    'val_macro_f1': float(f1_score(y_va, yhat, average = 'macro', zero_division = 0)),
                    'val_acc': float(accuracy_score(y_va, yhat)),
                    'val_balanced_acc': float(balanced_accuracy_score(y_va, yhat)),
                    'n_val': int(len(y_va)) }
                append_validation(val_row)
                return (new_rows, val_row)
                return (new_rows, val_row)
                except Exception:
                    e = None
                    new_rows.append({
                        'config_id': cfg['config_id'],
                        'model': cfg['model'],
                        'params_str': cfg['params_str'],
                        'feature_set': cfg['feature_set'],
                        'preproc': cfg['preproc'],
                        'fold': i,
                        'held_out_subject': int(subjects[te][0]),
                        'n_test': int(len(te)),
                        'acc': float('nan'),
                        'macro_f1': float('nan'),
                        'balanced_acc': float('nan'),
                        'elapsed_s': time.time() - t0 })
                    e = None
                    del e
                    continue
                    e = None
                    del e
            except Exception:
                val_row = None
                return (new_rows, val_row)


    
    def ensemble_top_k(summary = None, df_z = None, fs_map = None, k = (5,)):
        '''Average predicted probabilities of the top-K configs by LOSO macro-F1.
    Re-runs LOSO so we can collect per-fold probabilities; only top-K configs
    so this is fast.
    '''
        top = summary.head(k).copy()
        print(f'''[ensemble] using top-{k} configs:''')
        for _, r in top.iterrows():
            print(f'''  - {r['model']:<10} fs={r['feature_set']:<14} pp={r['preproc']:<6} loso={r['loso_macro_f1_mean']:.3f}''')
        train = df_z[df_z['split'] == 'train'].reset_index(drop = True)
        val = df_z[df_z['split'] == 'validation'].reset_index(drop = True)
        train_ah = train[train['class'].isin(ARM_HAND)].reset_index(drop = True)
        val_ah = val[val['class'].isin(ARM_HAND)].reset_index(drop = True)
        subjects = train_ah['subject'].to_numpy()
        y_tr = train_ah['class'].map(LABEL_MAP).to_numpy()
        y_va = val_ah['class'].map(LABEL_MAP).to_numpy() if not val_ah.empty else None
        logo = LeaveOneGroupOut()
        n = len(train_ah)
        fold_idx = list(logo.split(np.zeros(n), y_tr, groups = subjects))
        proba_loso_per_cfg = []
        proba_val_per_cfg = []
    # WARNING: Decompyle incomplete

    
    def make_plots(summary = None):
        plt = pyplot
        import matplotlib.pyplot
        import seaborn as sns
        if summary.empty:
            return None
        top20 = summary.head(20).copy()
        top20['label'] = top20['model'] + ' | ' + top20['feature_set'] + ' | ' + top20['preproc']
        (fig, ax) = plt.subplots(figsize = (11, 8))
        ax.barh(top20['label'][::-1], top20['loso_macro_f1_mean'][::-1], xerr = top20['loso_macro_f1_std'][::-1], color = '#3b82f6', edgecolor = 'black')
        ax.axvline(0.5, color = 'r', linestyle = '--', label = 'chance')
        ax.set_xlabel('LOSO macro-F1')
        ax.set_title('Top-20 ARM-vs-HAND configurations')
        ax.legend()
        ax.set_xlim(0.4, 0.7)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'top20_loso_macro_f1.png', dpi = 130)
        plt.close(fig)
        if 'val_macro_f1' in summary.columns:
            sub = summary.dropna(subset = [
                'val_macro_f1']).copy()
            if not sub.empty:
                (fig, ax) = plt.subplots(figsize = (7, 6))
                ax.scatter(sub['loso_macro_f1_mean'], sub['val_macro_f1'], alpha = 0.5)
                ax.plot([
                    0.4,
                    0.7], [
                    0.4,
                    0.7], 'k--', alpha = 0.5)
                ax.axhline(0.5, color = 'r', linestyle = ':', alpha = 0.5)
                ax.axvline(0.5, color = 'r', linestyle = ':', alpha = 0.5)
                ax.set_xlabel('LOSO macro-F1')
                ax.set_ylabel('Validation macro-F1')
                ax.set_title('LOSO vs Validation macro-F1, all configs')
                fig.tight_layout()
                fig.savefig(PLOT_DIR / 'loso_vs_val_scatter.png', dpi = 130)
                plt.close(fig)
        (fig, ax) = plt.subplots(figsize = (8, 5))
        if 'val_macro_f1' in summary.columns:
            df_box = summary[[
                'model',
                'loso_macro_f1_mean',
                'val_macro_f1']].melt(id_vars = [
                'model'], var_name = 'metric', value_name = 'macro_f1')
        else:
            df_box = summary[[
                'model',
                'loso_macro_f1_mean']].rename(columns = {
                'loso_macro_f1_mean': 'macro_f1' }).assign(metric = 'loso')
        sns.boxplot(data = df_box, x = 'model', y = 'macro_f1', hue = 'metric', ax = ax)
        ax.axhline(0.5, color = 'r', linestyle = '--', label = 'chance')
        ax.set_title('Distribution of macro-F1 per model family')
        plt.xticks(rotation = 20)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'macro_f1_per_model_box.png', dpi = 130)
        plt.close(fig)

    
    def main(skip_ensemble = None, max_configs = None):
        t0 = time.time()
        print('[load] features ...')
        (df_all, feat_cols) = load_features()
        print(f'''[load] {len(feat_cols)} features × {len(df_all)} segments''')
        print('[normalise] subject-z ...')
        df_z = apply_subject_z(df_all, feat_cols)
        print('[fs] computing feature-set rankings ...')
        fs_map = feature_strategies(df_z[df_z['class'].isin(ARM_HAND)], feat_cols)
        for k, v in fs_map.items():
            print(f'''  fs={k:<14} n={len(v)}''')
        cfgs = all_configs()
    # WARNING: Decompyle incomplete

    if __name__ == '__main__':
        ap = argparse.ArgumentParser()
        ap.add_argument('--skip-ensemble', action = 'store_true')
        ap.add_argument('--max-configs', type = int, default = None, help = 'Limit number of configs (for smoke testing)')
        args = ap.parse_args()
        main(skip_ensemble = args.skip_ensemble, max_configs = args.max_configs)
        return None
    return None
except Exception:
    _HAS_XGB = False
    continue

