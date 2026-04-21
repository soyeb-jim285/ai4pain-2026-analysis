# Source Generated with Decompyle++
# File: 24_subject_constrained_decoding.cpython-312.pyc (Python 3.12)

'''Subject-level constrained decoding experiments for AI4Pain 2026.

This script tests the main decoding hypothesis that the known per-subject class
counts (12 NoPain, 12 PainArm, 12 PainHand) can be exploited at inference time.

It compares:
1. A two-stage pipeline:
   - Stage 1: NoPain vs Pain
   - Stage 2: PainArm vs PainHand
2. A direct 3-class model.

For both families, it evaluates:
1. Unconstrained per-segment decoding.
2. Exact-count constrained decoding per subject.

Outputs
-------
- results/tables/constrained_decoding_summary*.csv
- results/tables/constrained_decoding_per_subject*.csv
- results/tables/constrained_decoding_predictions*.parquet
- results/reports/24_subject_constrained_decoding_summary*.md

Usage
-----
    uv run python scripts/24_subject_constrained_decoding.py
    uv run python scripts/24_subject_constrained_decoding.py --quick
    uv run python scripts/24_subject_constrained_decoding.py         --feature-parquet results/tables/all_features_merged_1022.parquet
'''
from __future__ import annotations
import argparse
import math
import sys
import time
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
    warnings.filterwarnings('ignore')
    SEED = 42
    ANALYSIS = Path(__file__).resolve().parents[1]
    TAB_DIR = ANALYSIS / 'results' / 'tables'
    REPORT_DIR = ANALYSIS / 'results' / 'reports'
    TAB_DIR.mkdir(parents = True, exist_ok = True)
    REPORT_DIR.mkdir(parents = True, exist_ok = True)
    META_COLS = [
        'split',
        'subject',
        'class',
        'segment_idx',
        'segment_id']
    CLASS_ORDER_3 = [
        'NoPain',
        'PainArm',
        'PainHand']
    CLASS_ORDER_AH = [
        'PainArm',
        'PainHand']
    ARM_HAND = tuple(CLASS_ORDER_AH)
    
    def load_features(feature_fp = None):
        if not feature_fp:
            feature_fp
        fp = TAB_DIR / 'all_features_merged.parquet'
        if not fp.exists():
            raise SystemExit(f'''missing {fp}''')
        return (pd.read_parquet(fp), fp)

    
    def output_paths(tag = None):
        suffix = f'''_{tag}''' if tag else ''
        return {
            'metric_csv': TAB_DIR / f'''constrained_decoding_per_subject{suffix}.csv''',
            'summary_csv': TAB_DIR / f'''constrained_decoding_summary{suffix}.csv''',
            'pred_parquet': TAB_DIR / f'''constrained_decoding_predictions{suffix}.parquet''',
            'report_md': REPORT_DIR / f'''24_subject_constrained_decoding_summary{suffix}.md''' }

    
    def infer_output_tag(feature_fp = None, explicit_tag = None):
        pass
    # WARNING: Decompyle incomplete

    
    def prep_feature_matrix(df = None):
        pass
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

    
    def cliffs_delta(a = None, b = None):
        if a.size == 0 or b.size == 0:
            return 0
        diffs = a[(:, None)] - b[(None, :)]
        return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / (a.size * b.size))

    
    def stage2_feature_sets(train_df = None, feat_cols = None):
        pain = train_df[train_df['class'].isin(ARM_HAND)].reset_index(drop = True)
    # WARNING: Decompyle incomplete

    
    def make_scaler(name = None):
        if name == 'std':
            return StandardScaler()
        if None == 'robust':
            return RobustScaler()
        raise None(name)

    
    def make_binary_model(name = None):
        if name == 'logreg':
            return LogisticRegression(penalty = 'l2', C = 1, class_weight = 'balanced', max_iter = 4000, solver = 'lbfgs', n_jobs = 1, random_state = SEED)
        if None == 'rf':
            return RandomForestClassifier(n_estimators = 400, max_depth = None, n_jobs = -1, class_weight = 'balanced', random_state = SEED)
        if None == 'xgb':
            if not _HAS_XGB:
                raise RuntimeError('xgboost not installed')
            return XGBClassifier(n_estimators = 200, max_depth = 4, learning_rate = 0.08, max_bin = 128, tree_method = 'hist', objective = 'binary:logistic', eval_metric = 'logloss', random_state = SEED, n_jobs = 4, verbosity = 0)
        raise None(name)

    
    def make_multiclass_model(name = None):
        if name == 'logreg':
            return LogisticRegression(penalty = 'l2', C = 1, class_weight = 'balanced', max_iter = 4000, solver = 'lbfgs', n_jobs = 1, random_state = SEED)
        if None == 'rf':
            return RandomForestClassifier(n_estimators = 400, max_depth = None, n_jobs = -1, class_weight = 'balanced', random_state = SEED)
        if None == 'xgb':
            if not _HAS_XGB:
                raise RuntimeError('xgboost not installed')
            return XGBClassifier(n_estimators = 200, max_depth = 4, learning_rate = 0.08, max_bin = 128, tree_method = 'hist', objective = 'multi:softprob', num_class = 3, eval_metric = 'mlogloss', random_state = SEED, n_jobs = 4, verbosity = 0)
        raise None(name)

    
    def fit_predict_proba(train_df, test_df, feat_cols = None, labels = None, scaler_name = None, model_factory = ('train_df', 'pd.DataFrame', 'test_df', 'pd.DataFrame', 'feat_cols', 'list[str]', 'labels', 'np.ndarray', 'scaler_name', 'str', 'return', 'np.ndarray')):
        scaler = make_scaler(scaler_name)
        X_tr = train_df[feat_cols].to_numpy(dtype = np.float32)
        X_te = test_df[feat_cols].to_numpy(dtype = np.float32)
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        model = model_factory()
        model.fit(X_tr, labels)
        return model.predict_proba(X_te)

    
    def pain_binary(classes = None):
        return (classes != 'NoPain').astype(int).to_numpy()

    
    def armhand_binary(classes = None):
        lut = {
            'PainArm': 0,
            'PainHand': 1 }
        return classes.map(lut).to_numpy()

    
    def class_codes_3(classes = None):
        pass
    # WARNING: Decompyle incomplete

    
    def safe_log_probs(probs = None):
        return np.log(np.clip(probs, 1e-06, 1))

    
    def exact_count_decode(log_scores = None, target_counts = None):
        (n_seg, n_classes) = log_scores.shape
        if n_classes != len(target_counts):
            raise ValueError('score columns and target counts disagree')
        if sum(target_counts) != n_seg:
            raise ValueError('target counts must sum to number of segments')
        dp = {
            tuple([
                0] * n_classes): 0 }
        backptrs = []
        for i in range(n_seg):
            next_dp = { }
            next_back = { }
            for state, score in dp.items():
                for cls_idx in range(n_classes):
                    if state[cls_idx] >= target_counts[cls_idx]:
                        continue
                    new_state = list(state)
                    tuple(new_state) = range(n_classes)
                    new_score = score + float(log_scores[(i, cls_idx)])
                    if not new_score > next_dp.get(new_state_t, -(math.inf)):
                        continue
                    next_dp[new_state_t] = new_score
                    next_back[new_state_t] = (state, cls_idx)
            dp = next_dp
            backptrs.append(next_back)
        final_state = tuple(target_counts)
        if final_state not in dp:
            raise RuntimeError('exact-count decoding failed to find a feasible assignment')
        decoded = np.full(n_seg, -1, dtype = int)
        state = final_state
        for i in range(n_seg - 1, -1, -1):
            (prev_state, cls_idx) = backptrs[i][state]
            decoded[i] = cls_idx
            state = prev_state
        return decoded

    
    def two_stage_unconstrained(pain_probs = None, armhand_probs = None):
        pain_hat = (pain_probs[(:, 1)] >= 0.5).astype(int)
        arm_hat = (armhand_probs[(:, 0)] >= 0.5).astype(int)
        pred = np.zeros(len(pain_hat), dtype = int)
        pred[pain_hat == 0] = 0
        pred[(pain_hat == 1) & (arm_hat == 1)] = 1
        pred[(pain_hat == 1) & (arm_hat == 0)] = 2
        return pred

    
    def two_stage_constrained(pain_probs = None, armhand_probs = None):
        p_pain = np.clip(pain_probs[(:, 1)], 1e-06, 0.999999)
        p_arm = np.clip(armhand_probs[(:, 0)], 1e-06, 0.999999)
        log_scores = np.column_stack([
            np.log1p(-p_pain),
            np.log(p_pain) + np.log(p_arm),
            np.log(p_pain) + np.log1p(-p_arm)])
        return exact_count_decode(log_scores, [
            12,
            12,
            12])

    
    def direct_unconstrained(probs_3 = None):
        return probs_3.argmax(axis = 1).astype(int)

    
    def direct_constrained(probs_3 = None):
        return exact_count_decode(safe_log_probs(probs_3), [
            12,
            12,
            12])

    
    def metrics_row(y_true, y_pred, split = None, subject = None, pipeline = None, decode_mode = ('y_true', 'np.ndarray', 'y_pred', 'np.ndarray', 'split', 'str', 'subject', 'int', 'pipeline', 'str', 'decode_mode', 'str', 'return', 'dict')):
        return {
            'split': split,
            'subject': subject,
            'pipeline': pipeline,
            'decode_mode': decode_mode,
            'n': int(len(y_true)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average = 'macro', zero_division = 0)) }

    
    def add_prediction_rows(rows, df_subj, split, pipeline, decode_mode = None, pred_codes = None, pain_probs = None, armhand_probs = (None, None, None), probs_3 = ('rows', 'list[dict]', 'df_subj', 'pd.DataFrame', 'split', 'str', 'pipeline', 'str', 'decode_mode', 'str', 'pred_codes', 'np.ndarray', 'pain_probs', 'np.ndarray | None', 'armhand_probs', 'np.ndarray | None', 'probs_3', 'np.ndarray | None', 'return', 'None')):
        pass
    # WARNING: Decompyle incomplete

    
    def run_loso(df_z = None, feat_cols = None, quick = None):
        pass
    # WARNING: Decompyle incomplete

    
    def run_validation(df_z = None, feat_cols = None, quick = None):
        pass
    # WARNING: Decompyle incomplete

    
    def summarise_metrics(metric_df = None):
        summary = metric_df.groupby([
            'split',
            'pipeline',
            'decode_mode'], as_index = False).agg(macro_f1_mean = ('macro_f1', 'mean'), macro_f1_std = ('macro_f1', 'std'), accuracy_mean = ('accuracy', 'mean'), balanced_accuracy_mean = ('balanced_accuracy', 'mean'), n_subjects = ('subject', 'nunique')).sort_values([
            'split',
            'macro_f1_mean'], ascending = [
            True,
            False]).reset_index(drop = True)
        return summary

    
    def write_report(summary, elapsed_s = None, quick = None, feature_fp = None, out_fp = ('summary', 'pd.DataFrame', 'elapsed_s', 'float', 'quick', 'bool', 'feature_fp', 'Path', 'out_fp', 'Path', 'return', 'None')):
        lines = [
            '# 24 — Subject-level constrained decoding\n']
        lines.append(f'''- runtime: {elapsed_s:.1f}s''')
        lines.append(f'''- quick mode: {'yes' if quick else 'no'}''')
        lines.append(f'''- feature table: `{feature_fp}`''')
        lines.append('')
        for split in ('train_loso', 'validation'):
            sub = summary[summary['split'] == split].copy()
            if sub.empty:
                continue
            lines.append(f'''## {split}''')
            lines.append('')
            lines.append('| pipeline | decode | macro-F1 mean | macro-F1 std | acc | bal acc |')
            lines.append('|---|---|---:|---:|---:|---:|')
            for _, row in sub.iterrows():
                std = row['macro_f1_std'] if pd.notna(row['macro_f1_std']) else 0
                lines.append(f'''| {row['pipeline']} | {row['decode_mode']} | {row['macro_f1_mean']:.3f} | {std:.3f} | {row['accuracy_mean']:.3f} | {row['balanced_accuracy_mean']:.3f} |''')
            lines.append('')
            for pipeline in sub['pipeline'].unique():
                pair = sub[sub['pipeline'] == pipeline].set_index('decode_mode')
                if not {
                    'argmax',
                    'exact_counts'}.issubset(pair.index):
                    continue
                delta = pair.loc[('exact_counts', 'macro_f1_mean')] - pair.loc[('argmax', 'macro_f1_mean')]
                lines.append(f'''- `{split}` delta for `{pipeline}`: {delta:+.3f} macro-F1''')
            lines.append('')
        out_fp.write_text('\n'.join(lines))

    
    def main(quick = None, skip_validation = None, feature_parquet = None, output_tag = (False, False, None, None)):
        t0 = time.time()
        print('[load] features ...')
        (df, feature_fp) = load_features(feature_parquet)
        tag = infer_output_tag(feature_fp, output_tag)
        out_paths = output_paths(tag)
        (df_all, feat_cols) = prep_feature_matrix(df)
        df_z = apply_subject_z(df_all, feat_cols)
        print(f'''[load] {len(feat_cols)} features × {len(df_z)} segments after subject-z''')
        print(f'''[load] feature parquet: {feature_fp}''')
        if tag:
            print(f'''[save] output tag: {tag}''')
        print('[run] LOSO constrained decoding ...')
        (loso_metrics, loso_preds) = run_loso(df_z, feat_cols, quick = quick)
        metric_frames = [
            loso_metrics]
        pred_frames = [
            loso_preds]
        if not skip_validation:
            print('[run] validation constrained decoding ...')
            (val_metrics, val_preds) = run_validation(df_z, feat_cols, quick = quick)
            metric_frames.append(val_metrics)
            pred_frames.append(val_preds)
        metric_df = pd.concat(metric_frames, ignore_index = True)
        pred_df = pd.concat(pred_frames, ignore_index = True)
        summary = summarise_metrics(metric_df)
        metric_df.to_csv(out_paths['metric_csv'], index = False)
        summary.to_csv(out_paths['summary_csv'], index = False)
        pred_df.to_parquet(out_paths['pred_parquet'], index = False)
        write_report(summary, time.time() - t0, quick = quick, feature_fp = feature_fp, out_fp = out_paths['report_md'])
        print('\n[done] top rows:')
        print(summary.head(12).to_string(index = False))

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--quick', action = 'store_true', help = 'use lighter all-logreg models for a fast smoke test')
        parser.add_argument('--skip-validation', action = 'store_true', help = 'only run LOSO on the training split')
        parser.add_argument('--feature-parquet', type = Path, default = None, help = 'Use an alternate merged feature table instead of results/tables/all_features_merged.parquet')
        parser.add_argument('--output-tag', type = str, default = None, help = 'Optional suffix for output files; auto-inferred from feature parquet if omitted')
        args = parser.parse_args()
        main(quick = args.quick, skip_validation = args.skip_validation, feature_parquet = args.feature_parquet, output_tag = args.output_tag)
        return None
    return None
except Exception:
    _HAS_XGB = False
    continue

