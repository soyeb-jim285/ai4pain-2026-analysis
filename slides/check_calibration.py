"""Verify whether stage-1 calibration plot is broken or axis mislabeled."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
parq = ROOT / "results" / "tables" / "suite32_validation_predictions.parquet"
df = pd.read_parquet(parq)

print("columns:", list(df.columns))
print("shape:", df.shape)
print("unique true_y:", sorted(df["true_y"].unique()))

# Best stage-1 config from report.
best_cfg = "truncate1022|subject_robust|bvp_eda_core|xgb|base|anchor_center_l05"
sub = df[df["config_id"] == best_cfg].copy()
print(f"rows for best cfg: {len(sub)}")
if len(sub) == 0:
    # fall back to base config
    best_cfg = "truncate1022|subject_robust|bvp_eda_core|xgb|base"
    sub = df[df["config_id"] == best_cfg].copy()
    print(f"fallback cfg rows: {len(sub)}")

y = sub["true_y"].to_numpy()
p_nopain = sub["score_prob_nopain"].to_numpy()

print("\n--- label coding check ---")
print("mean(true_y): ", y.mean(), "  (expect ~0.667 if 1=Pain, else ~0.333)")
print("mean(p_nopain) where true_y==0:", p_nopain[y == 0].mean())
print("mean(p_nopain) where true_y==1:", p_nopain[y == 1].mean())
print("mean(p_nopain) where true_y==2:", p_nopain[y == 2].mean() if (y == 2).any() else "n/a")

# If true_y in {0,1,2} then we need to collapse Pain classes.
if set(np.unique(y)).issuperset({0, 1, 2}):
    y_bin = (y != 0).astype(int)  # 1 = Pain (PainArm or PainHand), 0 = NoPain
    print("\nCollapsed 3-class -> binary: 0=NoPain, 1=Pain")
else:
    y_bin = y.astype(int)

# Correct NoPain calibration: pos_label=0 => y_pos = (y_bin == 0)
y_is_nopain = (y_bin == 0).astype(int)

print("\n--- calibration: original plotting code (wrong) ---")
prob_true_wrong, prob_pred_wrong = calibration_curve(y_bin, p_nopain, n_bins=8, strategy="uniform")
for pp, pt in zip(prob_pred_wrong, prob_true_wrong):
    print(f"  pred={pp:.3f}  observed(Pain_as_pos)={pt:.3f}")

print("\n--- calibration: correct (pos_label=NoPain) ---")
prob_true_right, prob_pred_right = calibration_curve(y_is_nopain, p_nopain, n_bins=8, strategy="uniform")
for pp, pt in zip(prob_pred_right, prob_true_right):
    print(f"  pred={pp:.3f}  observed(NoPain)={pt:.3f}")

print("\n--- scalar metrics (NoPain positive) ---")
print(f"  Brier  = {brier_score_loss(y_is_nopain, p_nopain):.4f}")
print(f"  LogLoss= {log_loss(y_is_nopain, np.clip(p_nopain, 1e-6, 1-1e-6)):.4f}")

# Monotonicity check.
x = np.array(prob_pred_right)
t = np.array(prob_true_right)
dirn = np.sign(np.diff(t))
print(f"\nmonotone-up steps: {(dirn > 0).sum()} / {len(dirn)}")
print("overall trend:", "UP (calibrated)" if np.corrcoef(x, t)[0, 1] > 0.5 else "FLAT/INVERSE")
print("corr(pred,obs):", np.corrcoef(x, t)[0, 1])
