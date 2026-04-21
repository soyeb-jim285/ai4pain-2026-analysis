# Source Generated with Decompyle++
# File: 32_stage1_upgrade_suite.cpython-312.pyc (Python 3.12)

'''Stage-1 upgrade suite for AI4Pain 2026.

This suite focuses exclusively on improving binary NoPain-vs-Pain detection,
guided by the 2025 papers and the recent localisation experiments.

Search design
-------------
Base search: 72 configs
- 3 input modes: truncate1022, linear1022, poly1022
- 2 normalisations: subject_z, subject_robust
- 4 feature sets: bvp_only, eda_only, bvp_eda_core, bvp_eda_resp_small
- 3 models: xgb, rf, logreg

Refinement search: 24 configs
- top 6 base configs on validation
- 4 NoPain-anchor post-processing variants using the inferred 12 NoPain windows

Total full-pipeline configs: 96 (under the requested cap of 100).

Outputs
-------
- results/tables/suite32_base_summary.csv
- results/tables/suite32_refine_summary.csv
- results/tables/suite32_pipeline_summary.csv
- results/tables/suite32_per_subject.csv
- results/tables/suite32_validation_predictions.parquet
- results/tables/suite32_stage1_xgb_curves.csv
- results/reports/32_stage1_upgrade_suite.md
- plots/suite32/*.png

Usage
-----
    uv run python scripts/32_stage1_upgrade_suite.py
'''
from __future__ import annotations
import argparse
import importlib.util as importlib
import math
import time
from pathlib import Path
from matplotlib.pyplot import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
TAB_DIR = ROOT / 'results' / 'tables'
REPORT_DIR = ROOT / 'results' / 'reports'
PLOT_DIR = ROOT / 'plots' / 'suite32'
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR):
    d.mkdir(parents = True, exist_ok = True)

def load_module(script_name = None, module_name = None):
    path = ROOT / 'scripts' / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
# WARNING: Decompyle incomplete

BASE = load_module('24_subject_constrained_decoding.py', 'ai4pain_subject_constrained')
SWEEP = load_module('28_preproc_filter_sweep.py', 'ai4pain_preproc_filter')
BIN_NAMES = [
    'NoPain',
    'Pain']

def feature_table_specs():
    return [
        {
            'tag': 'truncate1022',
            'fp': TAB_DIR / 'all_features_merged_1022.parquet' },
        {
            'tag': 'linear1022',
            'fp': TAB_DIR / 'all_features_merged_linear1022.parquet' },
        {
            'tag': 'poly1022',
            'fp': TAB_DIR / 'all_features_merged_poly1022.parquet' }]


def ensure_feature_tables():
    missing = []
    for spec in feature_table_specs():
        if spec['fp'].exists():
            continue
        if spec['tag'] == 'truncate1022':
            raise SystemExit(f'''missing baseline feature table: {spec['fp']}''')
        missing.append(spec['tag'].replace('1022', ''))
    if missing:
        mod30 = load_module('30_build_resampled_feature_tables.py', 'ai4pain_resample_builder')
        mod30.main(methods = missing, target_len = 1022)
        return None


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


def load_feature_method(spec = None):
    (df, _) = BASE.load_features(spec['fp'])
    (df_all, feat_cols) = BASE.prep_feature_matrix(df)
    return {
        'tag': spec['tag'],
        'fp': spec['fp'],
        'df_all': df_all,
        'feat_cols': feat_cols,
        'norm_map': {
            'subject_z': BASE.apply_subject_z(df_all, feat_cols),
            'subject_robust': SWEEP.apply_subject_robust(df_all, feat_cols) } }


def rank_binary_features(train_df = None, feat_cols = None):
    tmp = train_df[[
        'subject',
        'class'] + feat_cols].copy()
    tmp['pain_bin'] = np.where(tmp['class'] == 'NoPain', 'NoPain', 'Pain')
    agg = tmp.groupby([
        'subject',
        'pain_bin'])[feat_cols].mean(numeric_only = True)
    nop = agg.xs('NoPain', level = 'pain_bin')
    pain = agg.xs('Pain', level = 'pain_bin')
    common = nop.index.intersection(pain.index)
    nop = nop.loc[common]
    pain = pain.loc[common]
    scores = { }
    for feat in feat_cols:
        a = nop[feat].to_numpy()
        b = pain[feat].to_numpy()
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() < 5:
            scores[feat] = 0
            continue
        scores[feat] = abs(BASE.cliffs_delta(a[valid], b[valid]))
    return pd.Series(scores).sort_values(ascending = False).index.tolist()


def stage1_feature_sets(train_df = None, feat_cols = None):
    pass
# WARNING: Decompyle incomplete


def exact12_binary_predictions(df_sub = None, probs_nopain = None):
    pred = np.ones(len(df_sub), dtype = int)
# WARNING: Decompyle incomplete


def metric_dict(y_true = None, y_pred = None):
    return {
        'accuracy': float(np.mean(y_true == y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average = 'macro', zero_division = 0)),
        'precision_nopain': float(precision_score(y_true, y_pred, pos_label = 0, zero_division = 0)),
        'recall_nopain': float(recall_score(y_true, y_pred, pos_label = 0, zero_division = 0)) }


def fit_stage1_probs(df_norm = None, feat_cols = None, model_name = None):
    train = df_norm[df_norm['split'] == 'train'].reset_index(drop = True)
    val = df_norm[df_norm['split'] == 'validation'].reset_index(drop = True)
    train_probs = np.zeros((len(train), 2), dtype = np.float32)
    for subject in sorted(train['subject'].unique()):
        tr = train[train['subject'] != subject].reset_index(drop = True)
        te = train[train['subject'] == subject].reset_index(drop = True)
        probs = BASE.fit_predict_proba(tr, te, feat_cols, BASE.pain_binary(tr['class']), scaler_name = 'std', model_factory = (lambda m = (model_name,): BASE.make_binary_model(m)))
        train_probs[train['subject'] == subject] = probs
    curve_df = pd.DataFrame()
    if model_name == 'xgb':
        X_tr = train[feat_cols].to_numpy(dtype = np.float32)
        X_va = val[feat_cols].to_numpy(dtype = np.float32)
        y_tr = BASE.pain_binary(train['class'])
        y_va = BASE.pain_binary(val['class'])
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        mdl = BASE.make_binary_model('xgb')
        mdl.fit(X_tr_s, y_tr, eval_set = [
            (X_tr_s, y_tr),
            (X_va_s, y_va)], verbose = False)
        ev = mdl.evals_result()
        curve_df = pd.DataFrame({
            'iter': np.arange(1, len(ev.get('validation_0', { }).get('logloss', [])) + 1),
            'train_logloss': ev.get('validation_0', { }).get('logloss', []),
            'val_logloss': ev.get('validation_1', { }).get('logloss', []) })
        val_probs = mdl.predict_proba(X_va_s)
    else:
        val_probs = BASE.fit_predict_proba(train, val, feat_cols, BASE.pain_binary(train['class']), scaler_name = 'std', model_factory = (lambda m = (model_name,): BASE.make_binary_model(m)))
    return (train_probs, val_probs.astype(np.float32), curve_df)


def infer_anchor_ids(df_sub = None, probs_nopain = None):
    out = { }
    for subject in sorted(df_sub['subject'].unique()):
        sub = df_sub[df_sub['subject'] == subject].reset_index(drop = True)
        p = probs_nopain[df_sub['subject'] == subject]
        order = np.argsort(-p)[:12]
        out[int(subject)] = sub.iloc[order]['segment_id'].tolist()
    return out


def anchor_refined_scores(df_sub, feat_df_global, feat_cols = None, probs_nopain = None, mode = None, lam = ('df_sub', 'pd.DataFrame', 'feat_df_global', 'pd.DataFrame', 'feat_cols', 'list[str]', 'probs_nopain', 'np.ndarray', 'mode', 'str', 'lam', 'float', 'return', 'np.ndarray')):
    sub_global = feat_df_global.loc[df_sub.index].reset_index(drop = True)
    out = np.zeros(len(df_sub), dtype = np.float32)
    anchor_map = infer_anchor_ids(df_sub.reset_index(drop = True), probs_nopain)
    for subject in sorted(df_sub['subject'].unique()):
        mask = (df_sub['subject'] == subject).to_numpy()
        sub = sub_global[mask].reset_index(drop = True)
        base_ids = set(anchor_map[int(subject)])
        base = sub['segment_id'].isin(base_ids)
        X = sub[feat_cols].to_numpy(dtype = np.float32)
        mu = sub.loc[(base, feat_cols)].mean(axis = 0).to_numpy(dtype = np.float32)
        if mode == 'center':
            dist = np.sqrt(np.mean((X - mu) ** 2, axis = 1))
        elif mode == 'z':
            sd = sub.loc[(base, feat_cols)].std(axis = 0, ddof = 0).to_numpy(dtype = np.float32)
            sd = np.where(sd > 1e-06, sd, 1)
            dist = np.mean(np.abs((X - mu) / sd), axis = 1)
        else:
            raise ValueError(mode)
        p = probs_nopain[mask].astype(np.float32)
        p_z = (p - p.mean()) / (p.std() + 1e-06)
        d_z = (dist - dist.mean()) / (dist.std() + 1e-06)
        out[mask] = p_z - lam * d_z
    return out


def feature_counts_for_config(feature_data = None):
    out = { }
    for ftag, fdata in feature_data.items():
        for norm_name, df_norm in fdata['norm_map'].items():
            train_df = df_norm[df_norm['split'] == 'train'].reset_index(drop = True)
            fs_map = stage1_feature_sets(train_df, fdata['feat_cols'])
            for fs_name, cols in fs_map.items():
                out[(ftag, norm_name, fs_name)] = len(cols)
    return out


def plot_confusion(cm = None, class_names = None, title = None, out_fp = ('cm', 'np.ndarray', 'class_names', 'list[str]', 'title', 'str', 'out_fp', 'Path', 'return', 'None')):
    cmn = cm.astype(float) / cm.sum(axis = 1, keepdims = True).clip(min = 1)
    (fig, ax) = plt.subplots(figsize = (4.5, 4))
    sns.heatmap(cmn, annot = True, fmt = '.2f', cmap = 'Blues', xticklabels = class_names, yticklabels = class_names, ax = ax, vmin = 0, vmax = 1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_fp, dpi = 140)
    plt.close(fig)


def make_plots(base_summary, refine_summary, pipeline_summary = None, per_subject = None, val_preds = None, curve_df = ('base_summary', 'pd.DataFrame', 'refine_summary', 'pd.DataFrame', 'pipeline_summary', 'pd.DataFrame', 'per_subject', 'pd.DataFrame', 'val_preds', 'pd.DataFrame', 'curve_df', 'pd.DataFrame', 'return', 'None')):
    val = pipeline_summary[pipeline_summary['split'] == 'validation'].copy()
    base_val = base_summary[base_summary['split'] == 'validation'].copy()
    if 'stage1_label' not in base_val.columns:
        base_val['stage1_label'] = base_val['resample_tag'] + '|' + base_val['feature_set'] + '|' + base_val['model_id']
    (fig, ax) = plt.subplots(figsize = (12, 6))
    top = val.head(20).copy()
    ax.bar(range(len(top)), top['macro_f1_mean'], color = sns.color_palette('viridis', len(top)))
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top['config_id'], rotation = 80, ha = 'right', fontsize = 7)
    ax.set_ylabel('Validation macro-F1')
    ax.set_title('Top stage-1 validation configs')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'top_validation_configs.png', dpi = 140)
    plt.close(fig)
    (fig, ax) = plt.subplots(figsize = (10, 5))
    sns.barplot(data = base_val, x = 'stage1_label', y = 'macro_f1_mean', hue = 'resample_tag', ax = ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    ax.set_ylabel('Validation macro-F1')
    ax.set_title('Base stage-1 configs')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'base_stage1_summary.png', dpi = 140)
    plt.close(fig)
    pivot = base_val.pivot_table(index = 'resample_tag', columns = 'feature_set', values = 'macro_f1_mean', aggfunc = 'max')
    (fig, ax) = plt.subplots(figsize = (8, 4))
    sns.heatmap(pivot, annot = True, fmt = '.3f', cmap = 'mako', ax = ax)
    ax.set_title('Best validation macro-F1 by resample x feature set')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'heatmap_resample_featureset.png', dpi = 140)
    plt.close(fig)
    pivot2 = base_val.pivot_table(index = 'model_id', columns = 'norm_id', values = 'macro_f1_mean', aggfunc = 'max')
    (fig, ax) = plt.subplots(figsize = (6, 4))
    sns.heatmap(pivot2, annot = True, fmt = '.3f', cmap = 'rocket', ax = ax)
    ax.set_title('Best validation macro-F1 by model x normalization')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'heatmap_model_norm.png', dpi = 140)
    plt.close(fig)
    if not refine_summary.empty:
        ref_val = refine_summary[refine_summary['split'] == 'validation'].copy()
        (fig, ax) = plt.subplots(figsize = (8, 4.5))
        sns.barplot(data = ref_val.sort_values('macro_f1_mean', ascending = False).head(12), x = 'refine_id', y = 'macro_f1_mean', hue = 'base_config_id', ax = ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
        ax.set_ylabel('Validation macro-F1')
        ax.set_title('Top NoPain-anchor refinements')
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'anchor_refinement_summary.png', dpi = 140)
        plt.close(fig)
    if not curve_df.empty:
        top_xgb = base_val[base_val['model_id'] == 'xgb'].head(6)['config_id'].tolist()
        sub = curve_df[curve_df['config_id'].isin(top_xgb)]
        n = len(top_xgb)
        cols = 3
        rows = math.ceil(max(n, 1) / cols)
        (fig, axes) = plt.subplots(rows, cols, figsize = (5 * cols, 3.5 * rows), squeeze = False)
        for ax, cfg in zip(axes.flat, top_xgb):
            cur = sub[sub['config_id'] == cfg]
            ax.plot(cur['iter'], cur['train_logloss'], label = 'train')
            ax.plot(cur['iter'], cur['val_logloss'], label = 'validation')
            ax.set_title(cfg)
            ax.set_xlabel('boosting round')
            ax.set_ylabel('logloss')
            ax.legend(fontsize = 8)
        for ax in axes.flat[n:]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(PLOT_DIR / 'xgb_training_curves.png', dpi = 140)
        plt.close(fig)
    best_cfgs = val.head(8)['config_id'].tolist()
    for rank, cfg in enumerate(best_cfgs, start = 1):
        sub = val_preds[val_preds['config_id'] == cfg].copy()
        y_true = sub['true_y'].to_numpy()
        y_pred = sub['pred_y'].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels = [
            0,
            1])
        plot_confusion(cm, BIN_NAMES, f'''Validation CM #{rank}\n{cfg}''', PLOT_DIR / f'''confusion_matrix_{rank}.png''')
        ps = per_subject[(per_subject['config_id'] == cfg) & (per_subject['split'] == 'validation')].sort_values('macro_f1')
        (fig, ax) = plt.subplots(figsize = (10, 4))
        ax.bar(ps['subject'].astype(str), ps['macro_f1'], color = 'steelblue')
        ax.axhline(ps['macro_f1'].mean(), color = 'red', linestyle = '--', label = f'''mean={ps['macro_f1'].mean():.3f}''')
        ax.set_title(f'''Validation per-subject macro-F1 #{rank}\n{cfg}''')
        ax.set_xlabel('subject')
        ax.set_ylabel('macro-F1')
        ax.tick_params(axis = 'x', rotation = 90, labelsize = 7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f'''per_subject_{rank}.png''', dpi = 140)
        plt.close(fig)
    best_cfg = val.iloc[0]['config_id']
    best = val_preds[val_preds['config_id'] == best_cfg].copy()
    (prob_true, prob_pred) = calibration_curve(best['true_y'].to_numpy(), best['score_prob_nopain'].to_numpy(), n_bins = 8, strategy = 'uniform')
    (fig, ax) = plt.subplots(figsize = (5, 4))
    ax.plot([
        0,
        1], [
        0,
        1], linestyle = '--', color = 'gray')
    ax.plot(prob_pred, prob_true, marker = 'o')
    ax.set_title(f'''Calibration\n{best_cfg}''')
    ax.set_xlabel('Predicted NoPain probability')
    ax.set_ylabel('Observed NoPain frequency')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'calibration_best.png', dpi = 140)
    plt.close(fig)
    (fig, ax) = plt.subplots(figsize = (6, 4))
    best = best.sort_values('score_final', ascending = False).reset_index(drop = True)
    ax.plot(best.index, best['score_final'], marker = 'o', label = 'final score')
    ax.axvline(11.5, linestyle = '--', color = 'red', label = 'top 12 cutoff')
    ax.set_title(f'''NoPain score ranking\n{best_cfg}''')
    ax.set_xlabel('Rank within validation windows')
    ax.set_ylabel('Final NoPain score')
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'score_ranking_best.png', dpi = 140)
    plt.close(fig)


def write_report(base_summary = None, refine_summary = None, pipeline_summary = None, elapsed_s = ('base_summary', 'pd.DataFrame', 'refine_summary', 'pd.DataFrame', 'pipeline_summary', 'pd.DataFrame', 'elapsed_s', 'float', 'return', 'None')):
    val = pipeline_summary[pipeline_summary['split'] == 'validation'].copy()
    base_val = base_summary[base_summary['split'] == 'validation'].copy()
    lines = [
        '# 32 — Stage-1 Upgrade Suite\n']
    lines.append(f'''- runtime: {elapsed_s:.1f}s''')
    lines.append('- base search: 72 configs')
    lines.append('- refinement search: 24 configs')
    lines.append('- total full pipeline configs: 96')
    lines.append('')
    best = val.iloc[0]
    lines.append('## Best Overall')
    lines.append('')
    for key in ('config_id', 'resample_tag', 'norm_id', 'feature_set', 'model_id', 'refine_id', 'macro_f1_mean', 'accuracy_mean', 'precision_nopain_mean', 'recall_nopain_mean'):
        lines.append(f'''- {key}: {best[key]}''')
    lines.append('')
    lines.append('## Top Validation Configs')
    lines.append('')
    lines.append('| config | macro-F1 | acc | prec NoPain | rec NoPain |')
    lines.append('|---|---:|---:|---:|---:|')
    for _, row in val.head(12).iterrows():
        lines.append(f'''| {row['config_id']} | {row['macro_f1_mean']:.3f} | {row['accuracy_mean']:.3f} | {row['precision_nopain_mean']:.3f} | {row['recall_nopain_mean']:.3f} |''')
    lines.append('')
    lines.append('## Best Base Configs')
    lines.append('')
    lines.append('| config | macro-F1 | acc | resample | norm | features | model |')
    lines.append('|---|---:|---:|---|---|---|---|')
    for _, row in base_val.head(12).iterrows():
        lines.append(f'''| {row['config_id']} | {row['macro_f1_mean']:.3f} | {row['accuracy_mean']:.3f} | {row['resample_tag']} | {row['norm_id']} | {row['feature_set']} | {row['model_id']} |''')
    lines.append('')
    if not refine_summary.empty:
        ref_val = refine_summary[refine_summary['split'] == 'validation'].copy()
        lines.append('## Best Anchor Refinements')
        lines.append('')
        lines.append('| base | refine | macro-F1 | acc | prec NoPain | rec NoPain |')
        lines.append('|---|---|---:|---:|---:|---:|')
        for _, row in ref_val.head(12).iterrows():
            lines.append(f'''| {row['base_config_id']} | {row['refine_id']} | {row['macro_f1_mean']:.3f} | {row['accuracy_mean']:.3f} | {row['precision_nopain_mean']:.3f} | {row['recall_nopain_mean']:.3f} |''')
    (REPORT_DIR / '32_stage1_upgrade_suite.md').write_text('\n'.join(lines))


def main():
    t0 = time.time()
    ensure_feature_tables()
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
    return None
