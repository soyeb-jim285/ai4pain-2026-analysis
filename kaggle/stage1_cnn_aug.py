"""Stage 1 CNN with signal augmentation: NoPain vs Pain.
Goal: reduce LOSO variance via synthetic signal augmentation.
"""
# %% [markdown]
# # Stage 1 CNN + Augmentation

# %%
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 7:
        DEVICE = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    else:
        print(f"GPU too old (sm_{cap[0]}{cap[1]}). Using CPU.")
print(f"Device: {DEVICE}, Torch: {torch.__version__}")

INPUT_DIR = None
for _p in ["/kaggle/input/datasets/soyebjim/ai4pain-2026-data", "/kaggle/input/ai4pain-2026-data", "cache"]:
    if Path(_p).exists() and (Path(_p) / "train_meta_1022.parquet").exists():
        INPUT_DIR = Path(_p)
        break
if INPUT_DIR is None:
    raise SystemExit("No input dir found")
print(f"INPUT_DIR: {INPUT_DIR}")

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("results/final/stage1_cnn_kaggle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
TAG = "1022"
tr_t = np.load(INPUT_DIR / f"train_tensor_{TAG}.npz")["tensor"]
va_t = np.load(INPUT_DIR / f"validation_tensor_{TAG}.npz")["tensor"]
tr_m = pd.read_parquet(INPUT_DIR / f"train_meta_{TAG}.parquet")
va_m = pd.read_parquet(INPUT_DIR / f"validation_meta_{TAG}.parquet")
print(f"Train: {tr_t.shape}, Val: {va_t.shape}")
print(f"Classes: {tr_m['class'].value_counts().to_dict()}")

# %% [markdown]
# ## Per-subject channel z with NaN guard

# %%
def per_subject_z(t, meta):
    out = t.copy().astype(np.float32)
    for subj in meta["subject"].unique():
        idx = meta[meta["subject"] == subj].index.to_numpy()
        for ch in range(t.shape[1]):
            x = t[idx, ch, :].reshape(-1)
            x = x[np.isfinite(x)]
            mu = float(x.mean()) if len(x) > 1 else 0.0
            sd = float(x.std()) if len(x) > 1 else 1.0
            sd = sd if sd > 1e-6 else 1.0
            out[idx, ch, :] = (t[idx, ch, :] - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

tr_t = per_subject_z(tr_t, tr_m)
va_t = per_subject_z(va_t, va_m)

# %% [markdown]
# ## Signal augmentation: time warp + freq mask + amplitude scale

# %%
def time_warp(x, rng, max_shift=50):
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return torch.roll(x, shifts=shift, dims=-1)

def amplitude_scale(x, rng, sigma=0.15):
    scale = 1.0 + rng.normal(0, sigma, (x.shape[0], x.shape[1], 1))
    scale = torch.from_numpy(scale.astype(np.float32)).to(x.device)
    return x * scale

def freq_mask(x, rng, max_width=10, n_masks=1):
    x = x.clone()
    B, C, T = x.shape
    for _ in range(n_masks):
        f0 = int(rng.integers(0, T - max_width))
        w = int(rng.integers(0, max_width))
        x[:, :, f0:f0 + w] = 0
    return x

def signal_mixup(x, y, rng, alpha=0.2):
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1 - lam)
    perm = torch.randperm(x.shape[0])
    mixed = lam * x + (1 - lam) * x[perm]
    return mixed, y, y[perm], lam

def augment_batch(x, rng):
    if rng.random() < 0.6:
        x = time_warp(x, rng)
    if rng.random() < 0.5:
        x = amplitude_scale(x, rng)
    if rng.random() < 0.3:
        x = freq_mask(x, rng, max_width=80)
    x = x + torch.randn_like(x) * 0.05
    return x

# %% [markdown]
# ## Small CNN (binary stage 1)

# %%
class Stage1CNN(nn.Module):
    def __init__(self, n_ch=4, dropout=0.4):
        super().__init__()
        self.c1 = nn.Conv1d(n_ch, 32, 7, padding=3)
        self.b1 = nn.BatchNorm1d(32)
        self.c2 = nn.Conv1d(32, 64, 5, padding=2)
        self.b2 = nn.BatchNorm1d(64)
        self.c3 = nn.Conv1d(64, 128, 3, padding=1)
        self.b3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.b1(self.c1(x))))
        x = self.pool(F.relu(self.b2(self.c2(x))))
        x = self.pool(F.relu(self.b3(self.c3(x))))
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

print(f"Params: {sum(p.numel() for p in Stage1CNN().parameters()):,}")

# %% [markdown]
# ## Training

# %%
def train_model(Xtr, ytr, Xva, yva, epochs=60, lr=5e-4, bs=32, patience=15, use_aug=True, use_mixup=True):
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).long()
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=bs, shuffle=True)
    model = Stage1CNN(n_ch=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float("inf")
    best_state = None
    counter = 0
    history = []
    rng = np.random.default_rng(SEED)

    for ep in range(epochs):
        model.train(True)
        tr_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if use_aug:
                xb = augment_batch(xb, rng)
            if use_mixup and rng.random() < 0.5:
                xb, ya, yb2, lam = signal_mixup(xb, yb, rng)
                opt.zero_grad()
                logits = model(xb)
                loss = lam * F.cross_entropy(logits, ya) + (1 - lam) * F.cross_entropy(logits, yb2)
            else:
                opt.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        sched.step()
        tr_loss /= len(Xtr_t)

        va_loss = 0.0
        if Xva is not None and len(Xva) > 0:
            model.train(False)
            with torch.no_grad():
                xvt = torch.from_numpy(Xva).float().to(DEVICE)
                yvt = torch.from_numpy(yva).long().to(DEVICE)
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

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def predict(model, X, bs=64):
    model.train(False)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().to(DEVICE)
            logits = model(xb)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)

# %% [markdown]
# ## LOSO training on all train samples

# %%
y_tr_all = (tr_m["class"] != "NoPain").astype(int).to_numpy()
y_va_all = (va_m["class"] != "NoPain").astype(int).to_numpy()
print(f"Train: {len(tr_t)} | Val: {len(va_t)}")

# %%
t0 = time.time()
loso_pain_prob = np.zeros(len(tr_m), dtype=np.float32)
records = []
for subj in sorted(tr_m["subject"].unique()):
    tr_mask = (tr_m["subject"] != subj).to_numpy()
    te_mask = (tr_m["subject"] == subj).to_numpy()
    Xtr, ytr = tr_t[tr_mask], y_tr_all[tr_mask]
    Xte, yte = tr_t[te_mask], y_tr_all[te_mask]
    rng = np.random.default_rng(SEED + int(subj))
    idx = np.arange(len(Xtr)); rng.shuffle(idx)
    cut = int(0.9 * len(idx))
    model, _ = train_model(Xtr[idx[:cut]], ytr[idx[:cut]], Xtr[idx[cut:]], ytr[idx[cut:]], epochs=40, patience=12)
    probs = predict(model, Xte)
    loso_pain_prob[te_mask] = probs[:, 1]
    acc = float((probs.argmax(1) == yte).mean())
    records.append({"subject": int(subj), "acc": acc})
    if int(subj) % 5 == 0:
        print(f"  subj {subj}: acc={acc:.3f}")
print(f"LOSO time: {(time.time() - t0) / 60:.1f} min")
loso_df = pd.DataFrame(records)
loso_df.to_csv(OUT_DIR / "loso_history.csv", index=False)
loso_auc = roc_auc_score(y_tr_all, loso_pain_prob)
print(f"LOSO AUC: {loso_auc:.3f}")

# %% [markdown]
# ## Full train → validation

# %%
model_full, hist_full = train_model(tr_t, y_tr_all, va_t, y_va_all, epochs=60, patience=15)
val_pain_prob = predict(model_full, va_t)[:, 1]
train_pain_prob = predict(model_full, tr_t)[:, 1]
val_auc = roc_auc_score(y_va_all, val_pain_prob)
print(f"Val AUC: {val_auc:.3f}")

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(hist_full["epoch"], hist_full["train_loss"], label="train", linewidth=2)
ax.plot(hist_full["epoch"], hist_full["val_loss"], label="val", linewidth=2, linestyle="--")
ax.axhline(np.log(2), color="gray", linestyle=":", alpha=0.5, label="random")
ax.set_xlabel("epoch"); ax.set_ylabel("CE")
ax.set_title(f"Stage 1 CNN + aug (val AUC={val_auc:.3f})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT_DIR / "learning_curve.png", dpi=140); plt.show()

# %%
y_pred_va = (val_pain_prob > 0.5).astype(int)
cm = confusion_matrix(y_va_all, y_pred_va, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["NoPain", "Pain"], yticklabels=["NoPain", "Pain"], ax=ax, vmin=0, vmax=1)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Val CM AUC={val_auc:.3f}")
fig.tight_layout(); fig.savefig(OUT_DIR / "confusion_val.png", dpi=140); plt.show()

# %%
tr_m[["subject", "segment_id", "class"]].assign(pain_prob_cnn=loso_pain_prob).to_parquet(OUT_DIR / "loso_pain_probs.parquet", index=False)
va_m[["subject", "segment_id", "class"]].assign(pain_prob_cnn=val_pain_prob).to_parquet(OUT_DIR / "val_pain_probs.parquet", index=False)
tr_m[["subject", "segment_id", "class"]].assign(pain_prob_cnn=train_pain_prob).to_parquet(OUT_DIR / "train_full_pain_probs.parquet", index=False)
torch.save(model_full.state_dict(), OUT_DIR / "stage1_cnn.pt")

summary = {
    "loso_mean_acc": float(loso_df["acc"].mean()),
    "loso_std_acc": float(loso_df["acc"].std()),
    "loso_auc": float(loso_auc),
    "val_auc": float(val_auc),
}
print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
pd.Series(summary).to_csv(OUT_DIR / "summary.csv")
