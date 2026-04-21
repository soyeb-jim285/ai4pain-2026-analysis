"""Kaggle notebook: Stage 2 CNN for arm vs hand pain localization.

Upload via: jupytext --to ipynb stage2_cnn.py
Or paste cells directly into Kaggle kernel.

Dataset needed: soyebjim/ai4pain-2026-data
Expects:
  /kaggle/input/ai4pain-2026-data/cache/train_tensor_1022.npz
  /kaggle/input/ai4pain-2026-data/cache/train_meta_1022.parquet
  /kaggle/input/ai4pain-2026-data/cache/validation_tensor_1022.npz
  /kaggle/input/ai4pain-2026-data/cache/validation_meta_1022.parquet
"""

# %% [markdown]
# # Stage 2 CNN: Arm vs Hand Pain Localization
#
# Handcrafted RESP features give stage 2 AUC = 0.57 (near random).
# CNN trains on raw 4-channel waveforms to extract patterns handcrafted features miss.

# %%
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 7:
        DEVICE = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    else:
        print(f"GPU too old (sm_{cap[0]}{cap[1]}). Using CPU.")
else:
    print("No CUDA. Using CPU.")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}, Torch: {torch.__version__}")

INPUT_DIR = None
for _p in ["/kaggle/input/datasets/soyebjim/ai4pain-2026-data", "/kaggle/input/ai4pain-2026-data", "cache"]:
    if Path(_p).exists() and (Path(_p) / "train_meta_1022.parquet").exists():
        INPUT_DIR = Path(_p)
        break
if INPUT_DIR is None:
    raise SystemExit("No input dir found")
print(f"INPUT_DIR: {INPUT_DIR}")
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("results/final/stage2_cnn_kaggle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load data

# %%
TAG = "1022"
tr_t = np.load(INPUT_DIR / f"train_tensor_{TAG}.npz")["tensor"]
va_t = np.load(INPUT_DIR / f"validation_tensor_{TAG}.npz")["tensor"]
tr_m = pd.read_parquet(INPUT_DIR / f"train_meta_{TAG}.parquet")
va_m = pd.read_parquet(INPUT_DIR / f"validation_meta_{TAG}.parquet")
print(f"Train: {tr_t.shape} | {tr_m['subject'].nunique()} subjects")
print(f"Val:   {va_t.shape} | {va_m['subject'].nunique()} subjects")
print(f"Classes: {tr_m['class'].value_counts().to_dict()}")

# %% [markdown]
# ## Per-subject channel z-score

# %%
def per_subject_channel_z(t, meta):
    t_out = t.copy().astype(np.float32)
    for subj in meta["subject"].unique():
        idx = meta[meta["subject"] == subj].index.to_numpy()
        for ch in range(t.shape[1]):
            x = t[idx, ch, :].reshape(-1)
            x = x[np.isfinite(x)]
            if len(x) < 2:
                mu, sd = 0.0, 1.0
            else:
                mu, sd = float(x.mean()), float(x.std())
                sd = sd if sd > 1e-6 else 1.0
            t_out[idx, ch, :] = (t[idx, ch, :] - mu) / sd
    # Replace any remaining NaN/Inf
    t_out = np.nan_to_num(t_out, nan=0.0, posinf=0.0, neginf=0.0)
    return t_out

tr_t = per_subject_channel_z(tr_t, tr_m)
va_t = per_subject_channel_z(va_t, va_m)
print(f"tr_t finite: {np.isfinite(tr_t).all()}, range: [{tr_t.min():.3f}, {tr_t.max():.3f}]")
print(f"va_t finite: {np.isfinite(va_t).all()}, range: [{va_t.min():.3f}, {va_t.max():.3f}]")

# %% [markdown]
# ## CNN

# %%
class Stage2CNN(nn.Module):
    def __init__(self, n_channels=4, n_classes=2, dropout=0.35):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f"CNN params: {n_params(Stage2CNN()):,}")

# %% [markdown]
# ## Augmentation

# %%
def augment(x, rng):
    x = x.clone()
    B, C, T = x.shape
    for i in range(B):
        shift = int(rng.integers(-30, 30))
        if shift != 0:
            x[i] = torch.roll(x[i], shifts=shift, dims=-1)
    if rng.random() < 0.3:
        ch = int(rng.integers(0, C))
        x[:, ch, :] = 0
    x = x + torch.randn_like(x) * 0.05
    return x

# %% [markdown]
# ## Train fn (uses .train()/.__eval__())

# %%
def set_eval(model):
    model.train(False)

def set_train(model):
    model.train(True)

def train_cnn(X_tr, y_tr, X_va=None, y_va=None, epochs=50, lr=1e-3, bs=32, patience=10, use_aug=True, verbose=False):
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).long()
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)
    model = Stage2CNN(n_channels=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_state = None
    counter = 0
    history = []
    rng = np.random.default_rng(SEED)

    for ep in range(epochs):
        set_train(model)
        tr_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if use_aug:
                xb = augment(xb, rng)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        sched.step()
        tr_loss /= len(X_tr_t)

        va_loss = 0.0
        if X_va is not None and len(X_va) > 0:
            set_eval(model)
            with torch.no_grad():
                xvt = torch.from_numpy(X_va).float().to(DEVICE)
                yvt = torch.from_numpy(y_va).long().to(DEVICE)
                va_loss = F.cross_entropy(model(xvt), yvt).item()
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                break
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        if verbose and ep % 5 == 0:
            print(f"    ep{ep:3d}: tr={tr_loss:.4f} va={va_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def predict(model, X, bs=64):
    set_eval(model)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().to(DEVICE)
            logits = model(xb)
            # Replace NaN logits with zeros so softmax = uniform
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)

# %% [markdown]
# ## LOSO on pain windows

# %%
ARM_HAND = {"PainArm", "PainHand"}
pain_mask_tr = tr_m["class"].isin(ARM_HAND).to_numpy()
pain_idx_tr = np.flatnonzero(pain_mask_tr)
pain_meta_tr = tr_m.iloc[pain_idx_tr].reset_index(drop=True)
pain_t_tr = tr_t[pain_idx_tr]
y_tr_all = (pain_meta_tr["class"] == "PainHand").astype(int).to_numpy()
print(f"Pain windows train: {len(pain_meta_tr)}")

# %%
t0 = time.time()
loso_arm_prob = np.zeros(len(pain_meta_tr), dtype=np.float32)
records = []
for subj in sorted(pain_meta_tr["subject"].unique()):
    tr_mask = (pain_meta_tr["subject"] != subj).to_numpy()
    te_mask = (pain_meta_tr["subject"] == subj).to_numpy()
    Xtr = pain_t_tr[tr_mask]; ytr = y_tr_all[tr_mask]
    Xte = pain_t_tr[te_mask]; yte = y_tr_all[te_mask]
    rng = np.random.default_rng(SEED + int(subj))
    idx = np.arange(len(Xtr)); rng.shuffle(idx)
    cut = int(0.9 * len(idx))
    model, _ = train_cnn(Xtr[idx[:cut]], ytr[idx[:cut]], Xtr[idx[cut:]], ytr[idx[cut:]], epochs=40, patience=12, lr=5e-4, use_aug=False)
    probs = predict(model, Xte)
    loso_arm_prob[te_mask] = probs[:, 0]
    acc = float((probs.argmax(1) == yte).mean())
    records.append({"subject": int(subj), "acc": acc})
    if int(subj) % 5 == 0:
        print(f"  subj {subj}: acc={acc:.3f}")
print(f"LOSO time: {(time.time() - t0) / 60:.1f} min")
loso_df = pd.DataFrame(records)
loso_df.to_csv(OUT_DIR / "loso_history.csv", index=False)
loso_auc = roc_auc_score(1 - y_tr_all, loso_arm_prob)
print(f"LOSO AUC (arm positive): {loso_auc:.3f}")

# %% [markdown]
# ## Full-train + validation

# %%
pain_mask_va = va_m["class"].isin(ARM_HAND).to_numpy()
pain_idx_va = np.flatnonzero(pain_mask_va)
pain_meta_va = va_m.iloc[pain_idx_va].reset_index(drop=True)
pain_t_va = va_t[pain_idx_va]
y_va_all = (pain_meta_va["class"] == "PainHand").astype(int).to_numpy()

model_full, hist_full = train_cnn(pain_t_tr, y_tr_all, pain_t_va, y_va_all,
                                   epochs=60, patience=15, lr=5e-4, use_aug=False, verbose=True)
val_arm_prob = predict(model_full, pain_t_va)[:, 0]
train_arm_prob = predict(model_full, pain_t_tr)[:, 0]
val_auc = roc_auc_score(1 - y_va_all, val_arm_prob)
print(f"Val AUC: {val_auc:.3f}")

# %% [markdown]
# ## Plots

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(hist_full["epoch"], hist_full["train_loss"], label="train", linewidth=2)
ax.plot(hist_full["epoch"], hist_full["val_loss"], label="validation", linewidth=2, linestyle="--")
ax.axhline(np.log(2), color="gray", linestyle=":", alpha=0.5, label="random")
ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy")
ax.set_title(f"Stage 2 CNN (val AUC={val_auc:.3f})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "learning_curve.png", dpi=140)
plt.show()

y_pred_va = (val_arm_prob < 0.5).astype(int)
cm = confusion_matrix(y_va_all, y_pred_va, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Arm", "Hand"], yticklabels=["Arm", "Hand"], ax=ax, vmin=0, vmax=1)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Val CM  |  AUC={val_auc:.3f}")
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_val.png", dpi=140)
plt.show()

# %% [markdown]
# ## Save outputs

# %%
pain_meta_tr[["subject", "segment_id", "class"]].assign(arm_prob_cnn=loso_arm_prob).to_parquet(OUT_DIR / "loso_arm_probs.parquet", index=False)
pain_meta_va[["subject", "segment_id", "class"]].assign(arm_prob_cnn=val_arm_prob).to_parquet(OUT_DIR / "val_arm_probs.parquet", index=False)
pain_meta_tr[["subject", "segment_id", "class"]].assign(arm_prob_cnn=train_arm_prob).to_parquet(OUT_DIR / "train_full_arm_probs.parquet", index=False)
torch.save(model_full.state_dict(), OUT_DIR / "stage2_cnn.pt")

summary = {
    "loso_mean_acc": float(loso_df["acc"].mean()),
    "loso_std_acc": float(loso_df["acc"].std()),
    "loso_auc": float(loso_auc),
    "val_auc": float(val_auc),
    "n_params": int(n_params(model_full)),
}
print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
pd.Series(summary).to_csv(OUT_DIR / "summary.csv")
