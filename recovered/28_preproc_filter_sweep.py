# Source Generated with Decompyle++
# File: 28_preproc_filter_sweep.cpython-312.pyc (Python 3.12)

'''Preprocessing and feature-filter sweep around the best 1022 pipeline.

This script keeps the current best stage-1 model fixed and searches the stage-2
ARM-vs-HAND branch over:

1. Subject-level feature normalisation variants.
2. RESP-focused feature subsets.
3. Train-quantile clipping.
4. Final scaler choice.

Inference always uses the exact-count 12/12/12 decoder so we can optimise the
scores that matter most for the current winning setup.

Outputs
-------
- results/tables/preproc_filter_summary*.csv
- results/tables/preproc_filter_per_subject*.csv
- results/reports/28_preproc_filter_sweep_summary*.md

Usage
-----
    uv run python scripts/28_preproc_filter_sweep.py
    uv run python scripts/28_preproc_filter_sweep.py --quick
'''
from __future__ import annotations
import argparse
import importlib.util as importlib
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
TAB_DIR = ROOT / 'results' / 'tables'
REPORT_DIR = ROOT / 'results' / 'reports'
TAB_DIR.mkdir(parents = True, exist_ok = True)
REPORT_DIR.mkdir(parents = True, exist_ok = True)

def load_base_module(script_name = None, module_name = None):
    path = ROOT / 'scripts' / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
# WARNING: Decompyle incomplete

BASE = load_base_module('24_subject_constrained_decoding.py', 'ai4pain_subject_constrained')

def default_feature_parquet():
    fp_1022 = TAB_DIR / 'all_features_merged_1022.parquet'
    if fp_1022.exists():
        return fp_1022
    return None / 'all_features_merged.parquet'


def output_paths(tag = None):
    suffix = f'''_{tag}''' if tag else ''
    return {
        'summary_csv': TAB_DIR / f'''preproc_filter_summary{suffix}.csv''',
        'metric_csv': TAB_DIR / f'''preproc_filter_per_subject{suffix}.csv''',
        'report_md': REPORT_DIR / f'''28_preproc_filter_sweep_summary{suffix}.md''' }


def canonical_feature_name(name = None):
    if name.startswith('raw_'):
        return name[4:]


def unique_by_canonical(features = None):
    seen = set()
    out = []
    ordered = sorted(features, key = (lambda f: (f.startswith('raw_'), f)))
    for feat in ordered:
        canon = canonical_feature_name(feat)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(feat)
    return out


def apply_subject_robust(df = None, feat_cols = None):
    out = df.copy()
    med = out.groupby('subject')[feat_cols].transform('median')
    q75 = out.groupby('subject')[feat_cols].transform((lambda s: s.quantile(0.75)))
    q25 = out.groupby('subject')[feat_cols].transform((lambda s: s.quantile(0.25)))
    iqr = (q75 - q25).where(q75 - q25 > 0, 1)
    out[feat_cols] = ((out[feat_cols] - med) / iqr).fillna(0).astype(np.float32)
    return out


def apply_nopain_z(df = None, feat_cols = None):
    out = df.copy()
    for subject in out['subject'].unique():
        rows = out['subject'] == subject
        base = rows & (out['class'] == 'NoPain')
        if int(base.sum()) == 0:
            continue
        mu = out.loc[(base, feat_cols)].mean(axis = 0)
        sd = out.loc[(base, feat_cols)].std(axis = 0, ddof = 0)
        sd = sd.where(sd > 0, 1)
        out.loc[(rows, feat_cols)] = ((out.loc[(rows, feat_cols)] - mu) / sd).astype(np.float32)
    out[feat_cols] = out[feat_cols].fillna(0)
    return out


def build_normalized_variants(df_all = None, feat_cols = None):
    return {
        'subject_z': BASE.apply_subject_z(df_all, feat_cols),
        'subject_robust': apply_subject_robust(df_all, feat_cols),
        'nopain_z': apply_nopain_z(df_all, feat_cols) }


def rank_features(train_df = None, feat_cols = None):
    pain = train_df[train_df['class'].isin(BASE.ARM_HAND)].reset_index(drop = True)
    arm = pain[pain['class'] == 'PainArm'].groupby('subject')[feat_cols].mean()
    hand = pain[pain['class'] == 'PainHand'].groupby('subject')[feat_cols].mean()
    common = arm.index.intersection(hand.index)
    arm = arm.loc[common]
    hand = hand.loc[common]
    vals = { }
    for feat in feat_cols:
        a = arm[feat].to_numpy()
        h = hand[feat].to_numpy()
        valid = ~(np.isnan(a) | np.isnan(h))
        if valid.sum() < 5:
            vals[feat] = 0
            continue
        vals[feat] = abs(BASE.cliffs_delta(a[valid], h[valid]))
    return pd.Series(vals).sort_values(ascending = False)


def stage2_feature_sets(train_df = None, feat_cols = None):
    ranking = rank_features(train_df, feat_cols)
# WARNING: Decompyle incomplete


def clip_mode_apply(X_tr = None, X_te = None, clip_mode = None):
    if clip_mode == 'none':
        return (X_tr, X_te)
    if None == 'q01':
        lo = np.nanquantile(X_tr, 0.01, axis = 0)
        hi = np.nanquantile(X_tr, 0.99, axis = 0)
        return (np.clip(X_tr, lo, hi), np.clip(X_te, lo, hi))
    raise None(clip_mode)


def fit_stage2_one_fold(pain_tr, te_full = None, feat_cols = None, scaler_name = None, clip_mode = ('pain_tr', 'pd.DataFrame', 'te_full', 'pd.DataFrame', 'feat_cols', 'list[str]', 'scaler_name', 'str', 'clip_mode', 'str', 'return', 'np.ndarray')):
    X_tr = pain_tr[feat_cols].to_numpy(dtype = np.float32)
    y_tr = BASE.armhand_binary(pain_tr['class'])
    X_te = te_full[feat_cols].to_numpy(dtype = np.float32)
    (X_tr, X_te) = clip_mode_apply(X_tr, X_te, clip_mode)
    preproc = BASE.make_scaler(scaler_name)
    X_tr_t = preproc.fit_transform(X_tr)
    X_te_t = preproc.transform(X_te)
    mdl = BASE.make_binary_model('logreg')
    mdl.fit(X_tr_t, y_tr)
    return mdl.predict_proba(X_te_t)


def compute_stage1_probs(df_stage1 = None, feat_cols = None, quick = None):
    pass
# WARNING: Decompyle incomplete


def decode_subjectwise(df_sub = None, pain_probs = None, stage2_probs = None):
    rows = []
    for subject in sorted(df_sub['subject'].unique()):
        mask = (df_sub['subject'] == subject).to_numpy()
        pred = BASE.two_stage_constrained(pain_probs[mask], stage2_probs[mask])
        truth = BASE.class_codes_3(df_sub.loc[(mask, 'class')])
        rows.append(BASE.metrics_row(truth, pred, split = str(df_sub.loc[(mask, 'split')].iloc[0]), subject = int(subject), pipeline = '', decode_mode = 'exact_counts'))
    return rows


def config_grid(quick = None):
    feature_sets = [
        'resp_all',
        'resp_top10',
        'resp_top20_unique',
        'resp_bvp5']
    norms = [
        'subject_z',
        'subject_robust',
        'nopain_z']
    scalers = [
        'robust',
        'std']
    clips = [
        'none'] if quick else [
        'none',
        'q01']
    cfgs = []
    for norm in norms:
        for fs in feature_sets:
            for scaler in scalers:
                for clip in clips:
                    cfgs.append({
                        'norm': norm,
                        'feature_set': fs,
                        'scaler': scaler,
                        'clip': clip,
                        'config_id': f'''{norm}|{fs}|{scaler}|{clip}''' })
    return cfgs


def write_report(summary, feature_fp = None, out_fp = None, quick = None, elapsed_s = ('summary', 'pd.DataFrame', 'feature_fp', 'Path', 'out_fp', 'Path', 'quick', 'bool', 'elapsed_s', 'float', 'return', 'None')):
    lines = [
        '# 28 — Preprocessing and feature-filter sweep\n']
    lines.append(f'''- runtime: {elapsed_s:.1f}s''')
    lines.append(f'''- quick mode: {'yes' if quick else 'no'}''')
    lines.append(f'''- feature table: `{feature_fp}`''')
    lines.append('- stage 1 held fixed to the current best pain-vs-no-pain model')
    lines.append('- stage 2 always decoded with exact 12/12/12 counts')
    lines.append('')
    for split in ('train', 'validation'):
        sub = summary[summary['split'] == split].copy()
        if sub.empty:
            continue
        lines.append(f'''## {split}''')
        lines.append('')
        lines.append('| config | macro-F1 mean | macro-F1 std | acc | bal acc | n_features |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for _, row in sub.head(12).iterrows():
            std = row['macro_f1_std'] if pd.notna(row['macro_f1_std']) else 0
            lines.append(f'''| {row['config_id']} | {row['macro_f1_mean']:.3f} | {std:.3f} | {row['accuracy_mean']:.3f} | {row['balanced_accuracy_mean']:.3f} | {int(row['n_features'])} |''')
        lines.append('')
    out_fp.write_text('\n'.join(lines))


def main(quick = None, feature_parquet = None, output_tag = None):
    t0 = time.time()
    if not feature_parquet:
        feature_parquet
    feature_fp = default_feature_parquet()
    (df, resolved_fp) = BASE.load_features(feature_fp)
    tag = BASE.infer_output_tag(resolved_fp, output_tag)
    out = output_paths(tag)
    print('[load] features ...')
    (df_all, feat_cols) = BASE.prep_feature_matrix(df)
    norm_map = build_normalized_variants(df_all, feat_cols)
    stage1_df = norm_map['subject_z']
    train_stage1 = stage1_df[stage1_df['split'] == 'train'].reset_index(drop = True)
    val_stage1 = stage1_df[stage1_df['split'] == 'validation'].reset_index(drop = True)
    print('[stage1] computing fixed pain-vs-no-pain probabilities ...')
    (stage1_train_probs, stage1_val_probs) = compute_stage1_probs(stage1_df, feat_cols, quick = quick)
    metric_rows = []
    cfgs = config_grid(quick)
    pbar = tqdm(cfgs, desc = 'stage2 configs')
    for cfg in pbar:
        pbar.set_postfix({
            'norm': cfg['norm'],
            'fs': cfg['feature_set'],
            'sc': cfg['scaler'],
            'cl': cfg['clip'] })
        df_norm = norm_map[cfg['norm']]
        train_df = df_norm[df_norm['split'] == 'train'].reset_index(drop = True)
        val_df = df_norm[df_norm['split'] == 'validation'].reset_index(drop = True)
        fs_map = stage2_feature_sets(train_df, feat_cols)
        cols = fs_map[cfg['feature_set']]
        train_probs_stage2 = np.zeros((len(train_df), 2), dtype = np.float32)
        for subject in sorted(train_df['subject'].unique()):
            tr = train_df[train_df['subject'] != subject].reset_index(drop = True)
            te = train_df[train_df['subject'] == subject].reset_index(drop = True)
            pain_tr = tr[tr['class'].isin(BASE.ARM_HAND)].reset_index(drop = True)
            probs = fit_stage2_one_fold(pain_tr, te, cols, scaler_name = cfg['scaler'], clip_mode = cfg['clip'])
            train_probs_stage2[train_df['subject'] == subject] = probs
        pain_train_full = train_df[train_df['class'].isin(BASE.ARM_HAND)].reset_index(drop = True)
        val_probs_stage2 = fit_stage2_one_fold(pain_train_full, val_df, cols, scaler_name = cfg['scaler'], clip_mode = cfg['clip'])
        train_metrics = pd.DataFrame(decode_subjectwise(train_stage1, stage1_train_probs, train_probs_stage2))
        val_metrics = pd.DataFrame(decode_subjectwise(val_stage1, stage1_val_probs, val_probs_stage2))
        for split_name, mdf in (('train', train_metrics), ('validation', val_metrics)):
            metric_rows.append({
                'split': split_name,
                'config_id': cfg['config_id'],
                'norm': cfg['norm'],
                'feature_set': cfg['feature_set'],
                'scaler': cfg['scaler'],
                'clip': cfg['clip'],
                'n_features': int(len(cols)),
                'macro_f1_mean': float(mdf['macro_f1'].mean()),
                'macro_f1_std': float(mdf['macro_f1'].std()),
                'accuracy_mean': float(mdf['accuracy'].mean()),
                'balanced_accuracy_mean': float(mdf['balanced_accuracy'].mean()),
                'n_subjects': int(mdf['subject'].nunique()) })
    summary = pd.DataFrame(metric_rows).sort_values([
        'split',
        'macro_f1_mean'], ascending = [
        True,
        False]).reset_index(drop = True)
    summary.to_csv(out['summary_csv'], index = False)
    summary.to_csv(out['metric_csv'], index = False)
    write_report(summary, resolved_fp, out['report_md'], quick = quick, elapsed_s = time.time() - t0)
    print('\n[done] top validation rows:')
    print(summary[summary['split'] == 'validation'].head(12).to_string(index = False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action = 'store_true', help = 'smaller search')
    parser.add_argument('--feature-parquet', type = Path, default = None, help = 'alternate merged feature table')
    parser.add_argument('--output-tag', type = str, default = None, help = 'optional output suffix')
    args = parser.parse_args()
    main(quick = args.quick, feature_parquet = args.feature_parquet, output_tag = args.output_tag)
    return None
