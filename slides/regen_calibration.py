"""Regenerate stage-1 calibration plot with correct pos_label."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
parq = ROOT / "results" / "tables" / "suite32_validation_predictions.parquet"
df = pd.read_parquet(parq)

best_cfg = "truncate1022|subject_robust|bvp_eda_core|xgb|base|anchor_center_l05"
sub = df[df["config_id"] == best_cfg]
y_bin = sub["true_y"].to_numpy()
p_nopain = sub["score_prob_nopain"].to_numpy()
y_is_nopain = (y_bin == 0).astype(int)

pt, pp = calibration_curve(y_is_nopain, p_nopain, n_bins=8, strategy="uniform")
brier = brier_score_loss(y_is_nopain, p_nopain)
ll = log_loss(y_is_nopain, np.clip(p_nopain, 1e-6, 1 - 1e-6))

out = ROOT / "slides" / "calibration_stage1_fixed.png"
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
ax.plot(pp, pt, "o-", color="steelblue", label="observed")
ax.set_xlabel("Predicted P(NoPain)")
ax.set_ylabel("Observed NoPain frequency")
ax.set_title(f"Stage-1 calibration (fixed)\nBrier={brier:.3f}  LogLoss={ll:.3f}")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(out, dpi=140)
plt.close(fig)
print("wrote", out)
print(f"Brier={brier:.4f}  LogLoss={ll:.4f}  corr={np.corrcoef(pp, pt)[0,1]:.3f}")
