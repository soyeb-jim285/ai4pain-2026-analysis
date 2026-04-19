"""Deep-learning training for ARM-vs-HAND localisation in AI4Pain 2026.

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
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data_loader import load_split  # noqa: E402

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT_ROOT = Path(os.environ.get(
    "AI4PAIN_OUTPUT_DIR", str(Path(__file__).resolve().parents[1])
))
TAB_DIR = OUT_ROOT / "results" / "tables"
REPORT_DIR = OUT_ROOT / "results" / "reports"
PLOT_DIR = OUT_ROOT / "plots" / "armhand_dl"
CKPT_DIR = OUT_ROOT / "results" / "checkpoints"
for d in (TAB_DIR, REPORT_DIR, PLOT_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

PERFOLD_FP = TAB_DIR / "armhand_dl_perfold.csv"
SUMMARY_FP = TAB_DIR / "armhand_dl_summary.csv"
VALIDATION_FP = TAB_DIR / "armhand_dl_validation.csv"
SSL_CKPT = CKPT_DIR / "ssl_encoder.pt"
REPORT_FP = REPORT_DIR / "19_armhand_dl_summary.md"

ARM_HAND = ("PainArm", "PainHand")
LABEL_MAP = {"PainArm": 0, "PainHand": 1}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def subject_normalise(tensor: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Per-subject channel z-score using all 36 of subject's segments."""
    out = tensor.copy()
    for sid in meta["subject"].unique():
        rows = meta.index[meta["subject"] == sid]
        chunk = out[rows]
        for c in range(chunk.shape[1]):
            x = chunk[:, c, :]
            mu = float(np.nanmean(x))
            sd = float(np.nanstd(x))
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            out[rows, c, :] = ((chunk[:, c, :] - mu) / sd).astype(np.float32)
    return np.nan_to_num(out, nan=0.0)


def load_arrays() -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    train_t, train_m = load_split("train")
    val_t, val_m = load_split("validation")
    train_n = subject_normalise(train_t, train_m)
    val_n = subject_normalise(val_t, val_m)
    return train_n, train_m, val_n, val_m


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    """Small 1-D CNN trunk shared by supervised + SSL."""

    def __init__(self, in_ch: int = 4, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x):  # (B, 4, T)
        return self.conv(x)  # (B, out_dim)


class CNNClassifier(nn.Module):
    def __init__(self, encoder: CNNEncoder, n_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.out_dim, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class SSLDecoder(nn.Module):
    """Decoder that reconstructs the masked channel from the encoder's latent.
    The encoder produces a single (B, out_dim) vector; the decoder expands it
    back to (B, 1, T)."""

    def __init__(self, latent_dim: int = 128, T: int = 1000, hid: int = 64):
        super().__init__()
        self.T = T
        self.fc = nn.Linear(latent_dim, hid * (T // 8))
        self.up = nn.Sequential(
            nn.Unflatten(1, (hid, T // 8)),
            nn.ConvTranspose1d(hid, hid // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hid // 2, hid // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hid // 4, 1, 4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        return self.up(x)


# ---------------------------------------------------------------------------
# Self-supervised pretraining (masked-channel reconstruction)
# ---------------------------------------------------------------------------
def pretrain_ssl(
    full_tensor: np.ndarray, epochs: int = 40, batch_size: int = 64,
    lr: float = 1e-3, mask_prob: float = 0.5,
) -> CNNEncoder:
    """Train CNNEncoder + SSLDecoder to reconstruct one randomly-masked channel
    from the other three. Saves encoder weights to SSL_CKPT.
    """
    if SSL_CKPT.exists():
        print(f"[ssl] reusing checkpoint {SSL_CKPT}")
        encoder = CNNEncoder().to(DEVICE)
        encoder.load_state_dict(torch.load(SSL_CKPT, map_location=DEVICE))
        return encoder

    print(f"[ssl] pretraining on {len(full_tensor)} segments × 4 channels × "
          f"{full_tensor.shape[2]} samples")
    X = torch.from_numpy(full_tensor).float()
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                    drop_last=False)

    encoder = CNNEncoder().to(DEVICE)
    decoder = SSLDecoder(latent_dim=encoder.out_dim, T=X.shape[2]).to(DEVICE)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    lossf = nn.MSELoss()

    pbar = tqdm(range(epochs), desc="SSL pretrain")
    for ep in pbar:
        encoder.train(True)
        decoder.train(True)
        ep_loss = 0.0
        n_batches = 0
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            # randomly mask one of the 4 channels per sample
            mask_ch = torch.randint(0, 4, (xb.size(0),), device=DEVICE)
            x_in = xb.clone()
            for i, c in enumerate(mask_ch):
                x_in[i, c] = 0.0
            target = torch.stack(
                [xb[i, c] for i, c in enumerate(mask_ch)], dim=0
            ).unsqueeze(1)  # (B, 1, T)
            opt.zero_grad()
            z = encoder(x_in)
            recon = decoder(z)  # (B, 1, T)
            loss = lossf(recon, target)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        pbar.set_postfix({"loss": ep_loss / max(n_batches, 1)})

    torch.save(encoder.state_dict(), SSL_CKPT)
    print(f"[ssl] saved encoder -> {SSL_CKPT}")
    return encoder


# ---------------------------------------------------------------------------
# Supervised LOSO
# ---------------------------------------------------------------------------
def _train_single_model(
    Xt: "torch.Tensor", yt: "torch.Tensor",
    encoder_init: CNNEncoder | None, epochs: int, batch_size: int, lr: float,
    freeze_encoder: bool, seed: int,
) -> CNNClassifier:
    torch.manual_seed(seed)
    if encoder_init is None:
        enc = CNNEncoder().to(DEVICE)
    else:
        enc = CNNEncoder().to(DEVICE)
        enc.load_state_dict(encoder_init.state_dict())
    if freeze_encoder:
        for p in enc.parameters():
            p.requires_grad_(False)
    model = CNNClassifier(enc).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True,
                    num_workers=0, drop_last=False)
    for _ in range(epochs):
        model.train(True)
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def train_one_fold(
    Xn: np.ndarray, y: np.ndarray, tr_idx: np.ndarray, te_idx: np.ndarray,
    encoder_init: CNNEncoder | None, epochs: int, batch_size: int, lr: float,
    freeze_encoder: bool, n_seeds: int = 1,
) -> tuple[np.ndarray, dict]:
    """Train n_seeds models on the same fold, average softmax probs."""
    Xt = torch.from_numpy(Xn[tr_idx]).float()
    Xe = torch.from_numpy(Xn[te_idx]).float()
    yt = torch.from_numpy(y[tr_idx]).long()
    ye = torch.from_numpy(y[te_idx]).long()
    probs = np.zeros((len(te_idx), 2), dtype=np.float64)
    for s in range(n_seeds):
        seed = SEED + s * 1000 + 1
        model = _train_single_model(Xt, yt, encoder_init, epochs, batch_size,
                                    lr, freeze_encoder, seed=seed)
        model.train(False)
        with torch.no_grad():
            logits = model(Xe.to(DEVICE))
            p = torch.softmax(logits, dim=1).cpu().numpy()
        probs += p
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    probs /= n_seeds
    yhat = probs.argmax(axis=1)
    return yhat, {
        "acc": accuracy_score(ye.numpy(), yhat),
        "macro_f1": f1_score(ye.numpy(), yhat, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(ye.numpy(), yhat),
        "n_seeds": int(n_seeds),
    }


def run_loso(
    config: str, train_n: np.ndarray, train_m: pd.DataFrame,
    encoder_init: CNNEncoder | None, epochs: int, batch_size: int, lr: float,
    freeze_encoder: bool, n_seeds: int = 1,
) -> tuple[list[dict], np.ndarray]:
    mask = train_m["class"].isin(ARM_HAND).to_numpy()
    Xn = train_n[mask]
    meta = train_m[mask].reset_index(drop=True)
    y = meta["class"].map(LABEL_MAP).to_numpy().astype(np.int64)
    subjects = meta["subject"].to_numpy()
    logo = LeaveOneGroupOut()
    folds = list(logo.split(Xn, y, groups=subjects))

    completed_folds = _completed_folds(config)
    print(f"[{config}] {len(completed_folds)}/{len(folds)} folds already done")

    rows: list[dict] = []
    all_pred = np.full(len(y), -1, dtype=np.int64)

    for i, (tr, te) in enumerate(tqdm(folds, desc=f"{config} LOSO")):
        if i in completed_folds:
            continue
        t0 = time.time()
        yhat, met = train_one_fold(
            Xn, y, tr, te, encoder_init,
            epochs=epochs, batch_size=batch_size, lr=lr,
            freeze_encoder=freeze_encoder, n_seeds=n_seeds,
        )
        all_pred[te] = yhat
        row = {
            "config": config, "fold": i,
            "held_out_subject": int(subjects[te][0]),
            "n_test": int(len(te)),
            "acc": met["acc"], "macro_f1": met["macro_f1"],
            "balanced_acc": met["balanced_acc"],
            "elapsed_s": time.time() - t0,
        }
        rows.append(row)
        _append_fold_row(row)
    return rows, y


def evaluate_validation(
    config: str, train_n: np.ndarray, train_m: pd.DataFrame,
    val_n: np.ndarray, val_m: pd.DataFrame,
    encoder_init: CNNEncoder | None, epochs: int, batch_size: int, lr: float,
    freeze_encoder: bool, n_seeds: int = 1,
) -> dict:
    train_mask = train_m["class"].isin(ARM_HAND).to_numpy()
    val_mask = val_m["class"].isin(ARM_HAND).to_numpy()
    Xt = train_n[train_mask]
    Xv = val_n[val_mask]
    yt = train_m[train_mask]["class"].map(LABEL_MAP).to_numpy().astype(np.int64)
    yv = val_m[val_mask]["class"].map(LABEL_MAP).to_numpy().astype(np.int64)

    Xt_t = torch.from_numpy(Xt).float()
    yt_t = torch.from_numpy(yt).long()
    Xv_t = torch.from_numpy(Xv).float()
    probs = np.zeros((len(yv), 2), dtype=np.float64)
    for s in range(n_seeds):
        seed = SEED + s * 1000 + 1
        model = _train_single_model(Xt_t, yt_t, encoder_init,
                                    epochs, batch_size, lr,
                                    freeze_encoder, seed=seed)
        model.train(False)
        with torch.no_grad():
            p = torch.softmax(model(Xv_t.to(DEVICE)), dim=1).cpu().numpy()
        probs += p
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    probs /= n_seeds
    yhat = probs.argmax(axis=1)
    return {
        "config": config,
        "val_acc": float(accuracy_score(yv, yhat)),
        "val_macro_f1": float(f1_score(yv, yhat, average="macro", zero_division=0)),
        "val_balanced_acc": float(balanced_accuracy_score(yv, yhat)),
        "n_val": int(len(yv)),
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
PERFOLD_HEADER = [
    "config", "fold", "held_out_subject", "n_test",
    "acc", "macro_f1", "balanced_acc", "elapsed_s",
]


def _completed_folds(config: str) -> set[int]:
    if not PERFOLD_FP.exists():
        return set()
    df = pd.read_csv(PERFOLD_FP)
    sub = df[df["config"] == config]
    return set(int(x) for x in sub["fold"].unique())


def _read_fold_row(config: str, fold: int) -> dict | None:
    if not PERFOLD_FP.exists():
        return None
    df = pd.read_csv(PERFOLD_FP)
    s = df[(df["config"] == config) & (df["fold"] == fold)]
    return None if s.empty else s.iloc[0].to_dict()


def _append_fold_row(row: dict) -> None:
    df = pd.DataFrame([row])[PERFOLD_HEADER]
    write_header = not PERFOLD_FP.exists()
    df.to_csv(PERFOLD_FP, mode="a", header=write_header, index=False)


def _save_validation(row: dict) -> None:
    fp = VALIDATION_FP
    df = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    df = df[df["config"] != row["config"]] if "config" in df.columns else df
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(fp, index=False)


def _write_summary() -> pd.DataFrame:
    if not PERFOLD_FP.exists():
        return pd.DataFrame()
    pf = pd.read_csv(PERFOLD_FP)
    g = pf.groupby("config")
    s = g["macro_f1"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "loso_macro_f1_mean", "std": "loso_macro_f1_std",
                 "count": "n_folds"}
    )
    s["loso_acc_mean"] = g["acc"].mean()
    s["loso_balanced_acc_mean"] = g["balanced_acc"].mean()
    s["elapsed_s_total"] = g["elapsed_s"].sum()
    s = s.reset_index()
    if VALIDATION_FP.exists():
        v = pd.read_csv(VALIDATION_FP)
        s = s.merge(v, on="config", how="left")
    s = s.sort_values("loso_macro_f1_mean", ascending=False)
    s.to_csv(SUMMARY_FP, index=False)
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(epochs_sup: int = 25, epochs_ssl: int = 40, epochs_ft: int = 25,
         batch_size: int = 64, lr: float = 1e-3, n_seeds: int = 1) -> None:
    t0 = time.time()
    print(f"[device] {DEVICE}")
    if DEVICE == "cuda":
        print(f"[device] GPU={torch.cuda.get_device_name(0)} "
              f"n_gpu={torch.cuda.device_count()}")
    print(f"[ensemble] n_seeds per fold = {n_seeds}")

    print("[load] arrays ...")
    train_n, train_m, val_n, val_m = load_arrays()
    print(f"[load] train tensor={train_n.shape}, validation tensor={val_n.shape}")

    full_for_ssl = np.concatenate([train_n, val_n], axis=0)

    sup_label = f"supervised_cnn_s{n_seeds}" if n_seeds > 1 else "supervised_cnn"
    ssl_label = f"ssl_finetune_s{n_seeds}" if n_seeds > 1 else "ssl_finetune"

    # Config A: supervised CNN, no pretraining
    print(f"\n>>> Config A: {sup_label} (random init)")
    run_loso(sup_label, train_n, train_m, encoder_init=None,
             epochs=epochs_sup, batch_size=batch_size, lr=lr,
             freeze_encoder=False, n_seeds=n_seeds)
    val_a = evaluate_validation(sup_label, train_n, train_m, val_n, val_m,
                                encoder_init=None, epochs=epochs_sup,
                                batch_size=batch_size, lr=lr,
                                freeze_encoder=False, n_seeds=n_seeds)
    _save_validation(val_a)
    _write_summary()

    # Config B: SSL pretrain + fine-tune
    print(f"\n>>> Config B: {ssl_label} (SSL pretrain + fine-tune)")
    encoder = pretrain_ssl(full_for_ssl, epochs=epochs_ssl,
                           batch_size=batch_size, lr=lr)
    run_loso(ssl_label, train_n, train_m, encoder_init=encoder,
             epochs=epochs_ft, batch_size=batch_size, lr=lr,
             freeze_encoder=False, n_seeds=n_seeds)
    val_b = evaluate_validation(ssl_label, train_n, train_m, val_n, val_m,
                                encoder_init=encoder, epochs=epochs_ft,
                                batch_size=batch_size, lr=lr,
                                freeze_encoder=False, n_seeds=n_seeds)
    _save_validation(val_b)
    summary = _write_summary()

    # Report
    lines = ["# 19 — ARM vs HAND deep-learning training\n"]
    lines.append(f"- device: `{DEVICE}` ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")
    lines.append(f"- runtime: {time.time() - t0:.1f}s")
    lines.append("")
    if not summary.empty:
        lines.append("## LOSO macro-F1 (chance = 0.50)\n")
        lines.append(summary.to_markdown(index=False, floatfmt=".3f"))
    REPORT_FP.write_text("\n".join(lines))
    print(f"\n[save] {REPORT_FP}")
    print(f"[save] {SUMMARY_FP}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-supervised", type=int, default=25)
    ap.add_argument("--epochs-ssl", type=int, default=40)
    ap.add_argument("--epochs-finetune", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="Train N independently-seeded CNNs per fold; "
                         "average softmax probabilities for prediction.")
    args = ap.parse_args()
    main(epochs_sup=args.epochs_supervised,
         epochs_ssl=args.epochs_ssl, epochs_ft=args.epochs_finetune,
         batch_size=args.batch_size, lr=args.lr,
         n_seeds=args.n_seeds)
