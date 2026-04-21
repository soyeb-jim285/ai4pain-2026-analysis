"""Stage 1 CNN v2: bigger, multi-scale, attention pooling, seed ensemble."""
# %% [markdown]
# # Stage 1 CNN v2 — multi-scale + attention + ensemble

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
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 7:
        DEVICE = torch.device("cuda")
        print(f"CUDA: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    else:
        print(f"GPU sm_{cap[0]}{cap[1]} too old. CPU.")

INPUT_DIR = None
for p in ["/kaggle/input/datasets/soyebjim/ai4pain-2026-data", "/kaggle/input/ai4pain-2026-data", "cache"]:
    if Path(p).exists() and (Path(p) / "train_meta_1022.parquet").exists():
        INPUT_DIR = Path(p); break
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("results/final/stage1_cnn_v2_kaggle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
tr_t = np.load(INPUT_DIR / "train_tensor_1022.npz")["tensor"]
va_t = np.load(INPUT_DIR / "validation_tensor_1022.npz")["tensor"]
tr_m = pd.read_parquet(INPUT_DIR / "train_meta_1022.parquet")
va_m = pd.read_parquet(INPUT_DIR / "validation_meta_1022.parquet")

# %%
def per_subj_z(t, meta):
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

tr_t = per_subj_z(tr_t, tr_m)
va_t = per_subj_z(va_t, va_m)

# %% [markdown]
# ## Multi-scale CNN with attention pooling

# %%
class MultiScaleBlock(nn.Module):
    """Parallel kernels: captures short+long patterns simultaneously."""
    def __init__(self, in_ch, out_ch, kernels=(3, 7, 15)):
        super().__init__()
        per_k = out_ch // len(kernels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, per_k, k, padding=k // 2),
                nn.BatchNorm1d(per_k),
                nn.GELU(),
            ) for k in kernels
        ])
        self.merge = nn.Conv1d(per_k * len(kernels), out_ch, 1)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.merge(torch.cat(feats, dim=1))


class AttentionPool(nn.Module):
    """Learned weighted pooling across time."""
    def __init__(self, in_ch):
        super().__init__()
        self.attn = nn.Conv1d(in_ch, 1, 1)

    def forward(self, x):
        w = F.softmax(self.attn(x), dim=-1)
        return (x * w).sum(dim=-1)


class CNNv2(nn.Module):
    def __init__(self, n_ch=4, dropout=0.25):
        super().__init__()
        self.block1 = MultiScaleBlock(n_ch, 64, kernels=(5, 11, 21))
        self.block2 = MultiScaleBlock(64, 128, kernels=(3, 7, 15))
        self.block3 = MultiScaleBlock(128, 256, kernels=(3, 5, 7))
        self.pool = nn.MaxPool1d(4)
        self.attn_pool = AttentionPool(256)
        self.fc1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.gelu(self.block1(x)))
        x = self.pool(F.gelu(self.block2(x)))
        x = self.pool(F.gelu(self.block3(x)))
        x = self.attn_pool(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


print(f"Params: {sum(p.numel() for p in CNNv2().parameters()):,}")

# %% [markdown]
# ## Augmentation

# %%
def aug_batch(x, rng):
    if rng.random() < 0.6:
        shift = int(rng.integers(-50, 51))
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)
    if rng.random() < 0.5:
        scale = 1.0 + rng.normal(0, 0.15, (x.shape[0], x.shape[1], 1))
        x = x * torch.from_numpy(scale.astype(np.float32)).to(x.device)
    if rng.random() < 0.3:
        T = x.shape[-1]
        f0 = int(rng.integers(0, T - 100))
        w = int(rng.integers(10, 100))
        x = x.clone()
        x[:, :, f0:f0 + w] = 0
    x = x + torch.randn_like(x) * 0.05
    return x


def signal_mixup(x, y, rng, alpha=0.2):
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1 - lam)
    perm = torch.randperm(x.shape[0])
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


# %% [markdown]
# ## Training with SWA (stochastic weight averaging)

# %%
def train_one(Xtr, ytr, Xva, yva, seed, epochs=80, lr=3e-4, bs=32, patience=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long()), batch_size=bs, shuffle=True)
    model = CNNv2(n_ch=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    best_val = float("inf"); best_state = None; counter = 0
    history = []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(True)
        tr_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = aug_batch(xb, rng)
            if rng.random() < 0.5:
                xb, ya, yb2, lam = signal_mixup(xb, yb, rng)
                opt.zero_grad()
                logits = model(xb)
                loss = lam * F.cross_entropy(logits, ya) + (1 - lam) * F.cross_entropy(logits, yb2)
            else:
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        sched.step()
        tr_loss /= len(Xtr)

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
            logits = torch.nan_to_num(model(xb), nan=0.0)
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_ensemble(Xtr, ytr, Xva, yva, n_seeds=3, epochs=80):
    """Train N models with different seeds, return list + averaged probs on val."""
    models = []
    for i in range(n_seeds):
        m, _ = train_one(Xtr, ytr, Xva, yva, seed=SEED + i, epochs=epochs)
        models.append(m)
    return models


def predict_ensemble(models, X):
    return np.mean([predict(m, X) for m in models], axis=0)


# %% [markdown]
# ## LOSO with seed ensemble

# %%
y_tr_all = (tr_m["class"] != "NoPain").astype(int).to_numpy()
y_va_all = (va_m["class"] != "NoPain").astype(int).to_numpy()

# %%
t0 = time.time()
loso_pain = np.zeros(len(tr_m), dtype=np.float32)
records = []
N_SEEDS = 3
for subj in sorted(tr_m["subject"].unique()):
    tr_mask = (tr_m["subject"] != subj).to_numpy()
    te_mask = (tr_m["subject"] == subj).to_numpy()
    Xtr, ytr = tr_t[tr_mask], y_tr_all[tr_mask]
    Xte, yte = tr_t[te_mask], y_tr_all[te_mask]
    rng = np.random.default_rng(SEED + int(subj))
    idx = np.arange(len(Xtr)); rng.shuffle(idx)
    cut = int(0.9 * len(idx))
    models = train_ensemble(Xtr[idx[:cut]], ytr[idx[:cut]], Xtr[idx[cut:]], ytr[idx[cut:]], n_seeds=N_SEEDS, epochs=60)
    probs = predict_ensemble(models, Xte)
    loso_pain[te_mask] = probs[:, 1]
    acc = float((probs.argmax(1) == yte).mean())
    records.append({"subject": int(subj), "acc": acc})
    if int(subj) % 5 == 0:
        print(f"  subj {subj}: acc={acc:.3f}  elapsed={(time.time()-t0)/60:.1f}min")
print(f"LOSO total: {(time.time() - t0) / 60:.1f} min")
loso_auc = roc_auc_score(y_tr_all, loso_pain)
print(f"LOSO AUC: {loso_auc:.3f}")

# %% [markdown]
# ## Full train ensemble → val

# %%
models = train_ensemble(tr_t, y_tr_all, va_t, y_va_all, n_seeds=N_SEEDS, epochs=100)
val_pain = predict_ensemble(models, va_t)[:, 1]
train_pain_full = predict_ensemble(models, tr_t)[:, 1]
val_auc = roc_auc_score(y_va_all, val_pain)
print(f"Val AUC (ensemble): {val_auc:.3f}")

# %%
# Plot learning curve from last trained model
_, hist = train_one(tr_t, y_tr_all, va_t, y_va_all, seed=SEED + N_SEEDS, epochs=60)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(hist["epoch"], hist["train_loss"], label="train", linewidth=2)
ax.plot(hist["epoch"], hist["val_loss"], label="val", linewidth=2, linestyle="--")
ax.axhline(np.log(2), color="gray", linestyle=":", label="random")
ax.set_title(f"Stage 1 CNN v2 (val AUC ensemble={val_auc:.3f})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT_DIR / "learning_curve.png", dpi=140); plt.show()

# %%
y_pred_va = (val_pain > 0.5).astype(int)
cm = confusion_matrix(y_va_all, y_pred_va, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["NoPain", "Pain"], yticklabels=["NoPain", "Pain"], ax=ax, vmin=0, vmax=1)
ax.set_title(f"Val CM AUC={val_auc:.3f}")
fig.tight_layout(); fig.savefig(OUT_DIR / "confusion_val.png", dpi=140); plt.show()

# %%
tr_m[["subject", "segment_id", "class"]].assign(pain_prob_cnn=loso_pain).to_parquet(OUT_DIR / "loso_pain_probs.parquet", index=False)
va_m[["subject", "segment_id", "class"]].assign(pain_prob_cnn=val_pain).to_parquet(OUT_DIR / "val_pain_probs.parquet", index=False)
tr_m[["subject", "segment_id", "class"]].assign(pain_prob_cnn=train_pain_full).to_parquet(OUT_DIR / "train_full_pain_probs.parquet", index=False)

from sklearn.metrics import f1_score
loso_df = pd.DataFrame(records)
loso_df.to_csv(OUT_DIR / "loso_history.csv", index=False)
summary = {
    "loso_mean_acc": float(loso_df["acc"].mean()),
    "loso_std_acc": float(loso_df["acc"].std()),
    "loso_auc": float(loso_auc),
    "val_auc": float(val_auc),
    "n_seeds": N_SEEDS,
}
print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
pd.Series(summary).to_csv(OUT_DIR / "summary.csv")
