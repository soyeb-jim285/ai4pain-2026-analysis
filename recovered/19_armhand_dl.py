# Source Generated with Decompyle++
# File: 19_armhand_dl.cpython-312.pyc (Python 3.12)

'''Deep-learning training for ARM-vs-HAND localisation in AI4Pain 2026.

Two configs trained back-to-back, both with strict LOSO across 41 train
subjects + held-out validation on 12 subjects:

  A. supervised 1-D CNN  (no pretraining)
  B. self-supervised pretrain  (masked-channel reconstruction on the full
     1908 segments incl. NoPain)  +  fine-tune on ARM vs HAND

Designed to be resumable: per-fold results are appended to a CSV after every
LOSO fold, and the SSL encoder is checkpointed once and reused. A second run
of the script picks up exactly where it stopped.

Uses environment variables:
  AI4PAIN_ROOT        -> root containing train/ + validation/  (loader)
  AI4PAIN_CACHE       -> where cached tensors live  (loader)
  AI4PAIN_OUTPUT_DIR  -> where results / plots / checkpoints go  (defaults
                          to repo root, used so a Kaggle run can redirect to
                          /kaggle/working)
'''
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg', force = True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.nn import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.data_loader import load_split
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
OUT_ROOT = Path(os.environ.get('AI4PAIN_OUTPUT_DIR', str(Path(__file__).resolve().parents[1])))
TAB_DIR = OUT_ROOT / 'results' / 'tables'
REPORT_DIR = OUT_ROOT / 'results' / 'reports'
PLOT_DIR = OUT_ROOT / 'plots' / 'armhand_dl'
CKPT_DIR = OUT_ROOT / 'results' / 'checkpoints'
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR, CKPT_DIR):
    d.mkdir(parents = True, exist_ok = True)
PERFOLD_FP = TAB_DIR / 'armhand_dl_perfold.csv'
SUMMARY_FP = TAB_DIR / 'armhand_dl_summary.csv'
VALIDATION_FP = TAB_DIR / 'armhand_dl_validation.csv'
SSL_CKPT = CKPT_DIR / 'ssl_encoder.pt'
REPORT_FP = REPORT_DIR / '19_armhand_dl_summary.md'
ARM_HAND = ('PainArm', 'PainHand')
LABEL_MAP = {
    'PainArm': 0,
    'PainHand': 1 }
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def subject_normalise(tensor = None, meta = None):
    """Per-subject channel z-score using all 36 of subject's segments."""
    out = tensor.copy()
    for sid in meta['subject'].unique():
        rows = meta.index[meta['subject'] == sid]
        chunk = out[rows]
        for c in range(chunk.shape[1]):
            x = chunk[(:, c, :)]
            mu = float(np.nanmean(x))
            sd = float(np.nanstd(x))
            if np.isfinite(sd) or sd <= 0:
                sd = 1
            out[(rows, c, :)] = ((chunk[(:, c, :)] - mu) / sd).astype(np.float32)
    return np.nan_to_num(out, nan = 0)


def load_arrays():
    (train_t, train_m) = load_split('train')
    (val_t, val_m) = load_split('validation')
    train_n = subject_normalise(train_t, train_m)
    val_n = subject_normalise(val_t, val_m)
    return (train_n, train_m, val_n, val_m)


class CNNEncoder(nn.Module):
    pass
# WARNING: Decompyle incomplete


class CNNClassifier(nn.Module):
    pass
# WARNING: Decompyle incomplete


class SSLDecoder(nn.Module):
    pass
# WARNING: Decompyle incomplete


def pretrain_ssl(full_tensor = None, epochs = None, batch_size = None, lr = (40, 64, 0.001, 0.5), mask_prob = ('full_tensor', 'np.ndarray', 'epochs', 'int', 'batch_size', 'int', 'lr', 'float', 'mask_prob', 'float', 'return', 'CNNEncoder')):
    '''Train CNNEncoder + SSLDecoder to reconstruct one randomly-masked channel
    from the other three. Saves encoder weights to SSL_CKPT.
    '''
    if SSL_CKPT.exists():
        print(f'''[ssl] reusing checkpoint {SSL_CKPT}''')
        encoder = CNNEncoder().to(DEVICE)
        encoder.load_state_dict(torch.load(SSL_CKPT, map_location = DEVICE))
        return encoder
    None(f'''[ssl] pretraining on {len(full_tensor)} segments × 4 channels × {full_tensor.shape[2]} samples''')
    X = torch.from_numpy(full_tensor).float()
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = False)
    encoder = CNNEncoder().to(DEVICE)
    decoder = SSLDecoder(latent_dim = encoder.out_dim, T = X.shape[2]).to(DEVICE)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr = lr, weight_decay = 0.0001)
    lossf = nn.MSELoss()
    pbar = tqdm(range(epochs), desc = 'SSL pretrain')
# WARNING: Decompyle incomplete


def train_one_fold(Xn, y, tr_idx, te_idx, encoder_init, epochs = None, batch_size = None, lr = None, freeze_encoder = ('Xn', 'np.ndarray', 'y', 'np.ndarray', 'tr_idx', 'np.ndarray', 'te_idx', 'np.ndarray', 'encoder_init', 'CNNEncoder | None', 'epochs', 'int', 'batch_size', 'int', 'lr', 'float', 'freeze_encoder', 'bool', 'return', 'tuple[np.ndarray, dict]')):
    Xt = torch.from_numpy(Xn[tr_idx]).float()
    Xe = torch.from_numpy(Xn[te_idx]).float()
    yt = torch.from_numpy(y[tr_idx]).long()
    ye = torch.from_numpy(y[te_idx]).long()
    dl = DataLoader(TensorDataset(Xt, yt), batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = False)
# WARNING: Decompyle incomplete


def run_loso(config, train_n, train_m, encoder_init, epochs = None, batch_size = None, lr = None, freeze_encoder = ('config', 'str', 'train_n', 'np.ndarray', 'train_m', 'pd.DataFrame', 'encoder_init', 'CNNEncoder | None', 'epochs', 'int', 'batch_size', 'int', 'lr', 'float', 'freeze_encoder', 'bool', 'return', 'tuple[list[dict], np.ndarray]')):
    mask = train_m['class'].isin(ARM_HAND).to_numpy()
    Xn = train_n[mask]
    meta = train_m[mask].reset_index(drop = True)
    y = meta['class'].map(LABEL_MAP).to_numpy().astype(np.int64)
    subjects = meta['subject'].to_numpy()
    logo = LeaveOneGroupOut()
    folds = list(logo.split(Xn, y, groups = subjects))
    completed_folds = _completed_folds(config)
    print(f'''[{config}] {len(completed_folds)}/{len(folds)} folds already done''')
    rows = []
    all_pred = np.full(len(y), -1, dtype = np.int64)
# WARNING: Decompyle incomplete


def evaluate_validation(config, train_n, train_m, val_n, val_m, encoder_init, epochs = None, batch_size = None, lr = None, freeze_encoder = ('config', 'str', 'train_n', 'np.ndarray', 'train_m', 'pd.DataFrame', 'val_n', 'np.ndarray', 'val_m', 'pd.DataFrame', 'encoder_init', 'CNNEncoder | None', 'epochs', 'int', 'batch_size', 'int', 'lr', 'float', 'freeze_encoder', 'bool', 'return', 'dict')):
    train_mask = train_m['class'].isin(ARM_HAND).to_numpy()
    val_mask = val_m['class'].isin(ARM_HAND).to_numpy()
    Xt = train_n[train_mask]
    Xv = val_n[val_mask]
    yt = train_m[train_mask]['class'].map(LABEL_MAP).to_numpy().astype(np.int64)
    yv = val_m[val_mask]['class'].map(LABEL_MAP).to_numpy().astype(np.int64)
# WARNING: Decompyle incomplete

PERFOLD_HEADER = [
    'config',
    'fold',
    'held_out_subject',
    'n_test',
    'acc',
    'macro_f1',
    'balanced_acc',
    'elapsed_s']

def _completed_folds(config = None):
    if not PERFOLD_FP.exists():
        return set()
    df = None.read_csv(PERFOLD_FP)
    sub = df[df['config'] == config]
    return (lambda .0: pass# WARNING: Decompyle incomplete
)(sub['fold'].unique()())


def _read_fold_row(config = None, fold = None):
    if not PERFOLD_FP.exists():
        return None
    df = pd.read_csv(PERFOLD_FP)
    s = df[(df['config'] == config) & (df['fold'] == fold)]
    if s.empty:
        return None
    return None.iloc[0].to_dict()


def _append_fold_row(row = None):
    df = pd.DataFrame([
        row])[PERFOLD_HEADER]
    write_header = not PERFOLD_FP.exists()
    df.to_csv(PERFOLD_FP, mode = 'a', header = write_header, index = False)


def _save_validation(row = None):
    fp = VALIDATION_FP
    df = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    df = df[df['config'] != row['config']] if 'config' in df.columns else df
    df = pd.concat([
        df,
        pd.DataFrame([
            row])], ignore_index = True)
    df.to_csv(fp, index = False)


def _write_summary():
    if not PERFOLD_FP.exists():
        return pd.DataFrame()
    pf = None.read_csv(PERFOLD_FP)
    g = pf.groupby('config')
    s = g['macro_f1'].agg([
        'mean',
        'std',
        'count']).rename(columns = {
        'mean': 'loso_macro_f1_mean',
        'std': 'loso_macro_f1_std',
        'count': 'n_folds' })
    s['loso_acc_mean'] = g['acc'].mean()
    s['loso_balanced_acc_mean'] = g['balanced_acc'].mean()
    s['elapsed_s_total'] = g['elapsed_s'].sum()
    s = s.reset_index()
    if VALIDATION_FP.exists():
        v = pd.read_csv(VALIDATION_FP)
        s = s.merge(v, on = 'config', how = 'left')
    s = s.sort_values('loso_macro_f1_mean', ascending = False)
    s.to_csv(SUMMARY_FP, index = False)
    return s


def main(epochs_sup = None, epochs_ssl = None, epochs_ft = None, batch_size = (25, 40, 25, 64, 0.001), lr = ('epochs_sup', 'int', 'epochs_ssl', 'int', 'epochs_ft', 'int', 'batch_size', 'int', 'lr', 'float', 'return', 'None')):
    t0 = time.time()
    print(f'''[device] {DEVICE}''')
    if DEVICE == 'cuda':
        print(f'''[device] GPU={torch.cuda.get_device_name(0)} n_gpu={torch.cuda.device_count()}''')
    print('[load] arrays ...')
    (train_n, train_m, val_n, val_m) = load_arrays()
    print(f'''[load] train tensor={train_n.shape}, validation tensor={val_n.shape}''')
    full_for_ssl = np.concatenate([
        train_n,
        val_n], axis = 0)
    print('\n>>> Config A: supervised CNN (random init)')
    run_loso('supervised_cnn', train_n, train_m, encoder_init = None, epochs = epochs_sup, batch_size = batch_size, lr = lr, freeze_encoder = False)
    val_a = evaluate_validation('supervised_cnn', train_n, train_m, val_n, val_m, encoder_init = None, epochs = epochs_sup, batch_size = batch_size, lr = lr, freeze_encoder = False)
    _save_validation(val_a)
    _write_summary()
    print('\n>>> Config B: SSL pretrain + fine-tune')
    encoder = pretrain_ssl(full_for_ssl, epochs = epochs_ssl, batch_size = batch_size, lr = lr)
    run_loso('ssl_finetune', train_n, train_m, encoder_init = encoder, epochs = epochs_ft, batch_size = batch_size, lr = lr, freeze_encoder = False)
    val_b = evaluate_validation('ssl_finetune', train_n, train_m, val_n, val_m, encoder_init = encoder, epochs = epochs_ft, batch_size = batch_size, lr = lr, freeze_encoder = False)
    _save_validation(val_b)
    summary = _write_summary()
    lines = [
        '# 19 — ARM vs HAND deep-learning training\n']
    lines.append(f'''- device: `{DEVICE}` ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})''')
    lines.append(f'''- runtime: {time.time() - t0:.1f}s''')
    lines.append('')
    if not summary.empty:
        lines.append('## LOSO macro-F1 (chance = 0.50)\n')
        lines.append(summary.to_markdown(index = False, floatfmt = '.3f'))
    REPORT_FP.write_text('\n'.join(lines))
    print(f'''\n[save] {REPORT_FP}''')
    print(f'''[save] {SUMMARY_FP}''')
    print('Done.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs-supervised', type = int, default = 25)
    ap.add_argument('--epochs-ssl', type = int, default = 40)
    ap.add_argument('--epochs-finetune', type = int, default = 25)
    ap.add_argument('--batch-size', type = int, default = 64)
    ap.add_argument('--lr', type = float, default = 0.001)
    args = ap.parse_args()
    main(epochs_sup = args.epochs_supervised, epochs_ssl = args.epochs_ssl, epochs_ft = args.epochs_finetune, batch_size = args.batch_size, lr = args.lr)
    return None
