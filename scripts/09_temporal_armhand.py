"""Temporal (within-segment) dynamics for PainArm vs PainHand.

The question: do arm-vs-hand differences live in the *timing* of reactions
within the 10-s segment rather than the aggregate magnitude? Previous
pass (06_class_tests) found 0/243 aggregate features surviving FDR<0.05.
This script splits each 10-s segment into 5 non-overlapping 2-s windows,
computes per-window stats + trajectory summaries, then runs paired
within-subject statistics on ARM vs HAND.

Run:
    uv run python scripts/09_temporal_armhand.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as spsig
from scipy import stats as sp_stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import SFREQ, SIGNALS, load_split  # noqa: E402

warnings.filterwarnings("ignore")
np.random.seed(42)
plt.rcParams["figure.dpi"] = 120
sns.set_context("talk", font_scale=0.75)

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_ROOT / "results" / "tables"
PLOTS_DIR = ANALYSIS_ROOT / "plots" / "armhand_temporal"
REPORTS_DIR = ANALYSIS_ROOT / "results" / "reports"
for d in (TABLES_DIR, PLOTS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_WINDOWS = 5
WINDOW_LEN = 200  # 2 s @ 100 Hz
N_SAMPLES = 1000  # 10 s @ 100 Hz
WINDOW_CENTER_TIMES = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # seconds

# Band-power ranges per signal (Hz)
BANDS = {
    "Bvp": (0.15, 0.4),
    "Eda": (0.05, 0.5),
    "Resp": (0.1, 0.5),
    "SpO2": (0.0, 0.1),
}

META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strip_nan(x: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(x)
    if not mask.any():
        return np.array([], dtype=np.float32)
    last = int(np.flatnonzero(mask)[-1]) + 1
    x = x[:last]
    if np.isnan(x).any():
        idx = np.arange(len(x))
        good = ~np.isnan(x)
        if good.sum() < 2:
            return np.array([], dtype=np.float32)
        x = np.interp(idx, idx[good], x[good]).astype(np.float32)
    return x


def window_stats(x: np.ndarray, band: tuple[float, float]) -> dict[str, float]:
    """Compute stats for a single window (1D array, length WINDOW_LEN)."""
    if len(x) < 8 or np.std(x) == 0:
        return {
            "mean": float(np.mean(x)) if len(x) else np.nan,
            "std": 0.0 if len(x) else np.nan,
            "rms": np.nan, "range": np.nan, "skew": np.nan,
            "kurt": np.nan, "p2p": np.nan, "dom_freq": np.nan, "band_pow": np.nan,
        }
    m = float(np.mean(x))
    s = float(np.std(x))
    rms = float(np.sqrt(np.mean(x * x)))
    rng = float(np.max(x) - np.min(x))
    try:
        sk = float(sp_stats.skew(x, bias=False))
    except Exception:
        sk = np.nan
    try:
        kt = float(sp_stats.kurtosis(x, bias=False))
    except Exception:
        kt = np.nan
    p2p = rng
    try:
        f, pxx = spsig.welch(x, fs=SFREQ, nperseg=min(128, len(x)))
        if len(f) > 1:
            dom_freq = float(f[int(np.argmax(pxx))])
            lo, hi = band
            mask = (f >= lo) & (f <= hi)
            band_pow = float(np.trapezoid(pxx[mask], f[mask])) if mask.any() else np.nan
        else:
            dom_freq = np.nan
            band_pow = np.nan
    except Exception:
        dom_freq = np.nan
        band_pow = np.nan
    return {
        "mean": m, "std": s, "rms": rms, "range": rng, "skew": sk,
        "kurt": kt, "p2p": p2p, "dom_freq": dom_freq, "band_pow": band_pow,
    }


def trajectory_features(
    per_window: list[dict[str, float]],
) -> dict[str, float]:
    """From 5 per-window dicts, produce trajectory features."""
    means = np.array([w["mean"] for w in per_window], dtype=float)
    stds = np.array([w["std"] for w in per_window], dtype=float)
    ranges = np.array([w["range"] for w in per_window], dtype=float)

    out: dict[str, float] = {}

    # Slope of window-mean
    if np.all(np.isfinite(means)):
        slope_mean, _, _, _, _ = sp_stats.linregress(WINDOW_CENTER_TIMES, means)
        out["slope_mean"] = float(slope_mean)
    else:
        out["slope_mean"] = np.nan

    # Slope of window-std
    if np.all(np.isfinite(stds)):
        slope_std, _, _, _, _ = sp_stats.linregress(WINDOW_CENTER_TIMES, stds)
        out["slope_std"] = float(slope_std)
    else:
        out["slope_std"] = np.nan

    # Window of peak (argmax of window-mean)
    if np.all(np.isfinite(means)):
        argmax_idx = int(np.argmax(means))
        out["peak_window"] = float(argmax_idx + 1)  # 1..5
        out["time_to_peak_s"] = float(WINDOW_CENTER_TIMES[argmax_idx])
        # Categorical bins encoded as indicator
        out["peak_early"] = float(argmax_idx in (0, 1))
        out["peak_mid"] = float(argmax_idx == 2)
        out["peak_late"] = float(argmax_idx in (3, 4))
    else:
        out["peak_window"] = np.nan
        out["time_to_peak_s"] = np.nan
        out["peak_early"] = np.nan
        out["peak_mid"] = np.nan
        out["peak_late"] = np.nan

    # Early vs late contrast
    if np.all(np.isfinite(means)):
        out["early_late_contrast"] = float(np.mean(means[3:]) - np.mean(means[:2]))
    else:
        out["early_late_contrast"] = np.nan

    # Window of max variance (onset indicator)
    if np.all(np.isfinite(stds)):
        out["max_var_window"] = float(int(np.argmax(stds)) + 1)
    else:
        out["max_var_window"] = np.nan

    # Additional: per-window means & stds passed through as w{i}_mean / w{i}_std
    for i, w in enumerate(per_window, 1):
        out[f"w{i}_mean"] = w["mean"]
        out[f"w{i}_std"] = w["std"]
        out[f"w{i}_rms"] = w["rms"]
        out[f"w{i}_range"] = w["range"]
        out[f"w{i}_skew"] = w["skew"]
        out[f"w{i}_kurt"] = w["kurt"]
        out[f"w{i}_p2p"] = w["p2p"]
        out[f"w{i}_domfreq"] = w["dom_freq"]
        out[f"w{i}_bandpow"] = w["band_pow"]

    # Slope of range across windows
    if np.all(np.isfinite(ranges)):
        slope_range, _, _, _, _ = sp_stats.linregress(WINDOW_CENTER_TIMES, ranges)
        out["slope_range"] = float(slope_range)
    else:
        out["slope_range"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Step 1: build temporal feature matrix
# ---------------------------------------------------------------------------
def build_temporal_features(
    train_tensor: np.ndarray,
    train_meta: pd.DataFrame,
    val_tensor: np.ndarray,
    val_meta: pd.DataFrame,
    flatline_ids: set[str],
) -> pd.DataFrame:
    """Concatenate splits, compute trajectory features per (segment, signal).
    Include NoPain too (needed for subject-baseline delta) plus Pain segments.
    Returns wide dataframe with meta + feature columns.
    """
    tensor = np.concatenate([train_tensor, val_tensor], axis=0)
    meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Keep all 3 classes so we can compute NoPain baselines for subject-z.
    rows = []
    for row_idx in tqdm(range(len(meta)), desc="segments"):
        r = meta.iloc[row_idx]
        seg_id = r["segment_id"]
        out_row: dict = {c: r[c] for c in META_COLS}
        is_flatline_spo2 = seg_id in flatline_ids
        for s_i, sig in enumerate(SIGNALS):
            raw = tensor[row_idx, s_i]
            x = strip_nan(raw)
            if len(x) < 5 * WINDOW_LEN:
                # pad with last value or NaN-fill windows
                pad_n = 5 * WINDOW_LEN - len(x)
                if len(x) > 0:
                    x = np.concatenate([x, np.full(pad_n, x[-1], dtype=np.float32)])
                else:
                    x = np.full(5 * WINDOW_LEN, np.nan, dtype=np.float32)
            else:
                x = x[: 5 * WINDOW_LEN]
            per_window = []
            for w_i in range(N_WINDOWS):
                seg = x[w_i * WINDOW_LEN : (w_i + 1) * WINDOW_LEN]
                stats = window_stats(seg, BANDS[sig])
                per_window.append(stats)
            traj = trajectory_features(per_window)
            # If SpO2 flatline, mark SpO2 features NaN
            if sig == "SpO2" and is_flatline_spo2:
                traj = {k: np.nan for k in traj}
            for k, v in traj.items():
                out_row[f"{sig}_{k}"] = v
        rows.append(out_row)
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Step 2: subject-z delta using each subject's NoPain segments
# ---------------------------------------------------------------------------
def add_subject_delta(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Add <feat>__delta = value - subject's NoPain-mean value for that feat."""
    # Compute per-subject NoPain mean
    nopain = df[df["class"] == "NoPain"]
    subj_baselines = nopain.groupby("subject")[feat_cols].mean()
    # Broadcast
    df = df.copy()
    # Map each row's subject to its baseline
    baseline = df["subject"].map(lambda s: subj_baselines.loc[s] if s in subj_baselines.index else None)
    base_df = pd.DataFrame(list(baseline), index=df.index, columns=feat_cols)
    delta = df[feat_cols].values - base_df.values
    delta_df = pd.DataFrame(
        delta, index=df.index, columns=[f"{c}__delta" for c in feat_cols]
    )
    return pd.concat([df, delta_df], axis=1)


# ---------------------------------------------------------------------------
# Step 3: paired within-subject tests (ARM vs HAND)
# ---------------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return float(gt - lt) / (nx * ny)


def signal_from_feature(fname: str) -> str:
    for s in SIGNALS:
        if fname.startswith(s + "_"):
            return s
    return "?"


def run_arm_vs_hand_tests(
    df_train: pd.DataFrame, feat_cols: list[str]
) -> pd.DataFrame:
    """For each feature, run paired Wilcoxon and mixed-effects LMM."""
    arm = df_train[df_train["class"] == "PainArm"]
    hand = df_train[df_train["class"] == "PainHand"]
    # Subject-mean per class
    arm_mean = arm.groupby("subject")[feat_cols].mean()
    hand_mean = hand.groupby("subject")[feat_cols].mean()
    common_subj = arm_mean.index.intersection(hand_mean.index)
    arm_mean = arm_mean.loc[common_subj]
    hand_mean = hand_mean.loc[common_subj]

    rows = []
    ph_df = df_train[df_train["class"].isin(["PainArm", "PainHand"])].copy()
    ph_df["class_bin"] = (ph_df["class"] == "PainHand").astype(int)  # ARM=0, HAND=1
    for feat in tqdm(feat_cols, desc="tests"):
        a = arm_mean[feat].to_numpy()
        h = hand_mean[feat].to_numpy()
        mask = ~(np.isnan(a) | np.isnan(h))
        a_v = a[mask]
        h_v = h[mask]
        row = {
            "feature": feat,
            "signal": signal_from_feature(feat),
            "n_subjects_paired": int(mask.sum()),
        }
        try:
            if len(a_v) > 1 and np.any(a_v - h_v != 0):
                W, p = sp_stats.wilcoxon(a_v, h_v, zero_method="wilcox")
                row["paired_W"] = float(W)
                row["paired_p"] = float(p)
            else:
                row["paired_W"] = np.nan
                row["paired_p"] = np.nan
        except Exception:
            row["paired_W"] = np.nan
            row["paired_p"] = np.nan
        row["cliff_delta"] = cliffs_delta(a_v, h_v) if len(a_v) > 0 else np.nan
        # Sign consistency: fraction of subjects where ARM > HAND
        diffs = a_v - h_v
        if len(diffs) > 0:
            nz = diffs[diffs != 0]
            row["sign_consistency_frac"] = (
                float((nz > 0).mean()) if len(nz) > 0 else 0.5
            )
        else:
            row["sign_consistency_frac"] = np.nan

        # Mixed-effects LMM on segment-level data (ARM+HAND).
        # Try multiple optimizers; fall back gracefully.
        sub_ph = ph_df[["subject", "class_bin", feat]].dropna()
        sub_ph = sub_ph.rename(columns={feat: "_y"})
        if (
            len(sub_ph) > 20
            and sub_ph["subject"].nunique() > 2
            and sub_ph["_y"].std() > 0
        ):
            coef = np.nan
            pval = np.nan
            for method in ("powell", "bfgs", "lbfgs"):
                try:
                    md = MixedLM.from_formula(
                        "_y ~ class_bin", groups="subject", data=sub_ph,
                    )
                    fit = md.fit(method=method, reml=True, disp=False)
                    c = float(fit.fe_params.get("class_bin", np.nan))
                    p = float(fit.pvalues.get("class_bin", np.nan))
                    if np.isfinite(c) and np.isfinite(p):
                        coef, pval = c, p
                        break
                except Exception:
                    continue
            row["mixed_lmm_coef"] = coef
            row["mixed_lmm_p"] = pval
        else:
            row["mixed_lmm_coef"] = np.nan
            row["mixed_lmm_p"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    # BH FDR
    for p_col, out_col in [("paired_p", "paired_p_fdr"), ("mixed_lmm_p", "mixed_lmm_p_fdr")]:
        pvals = out[p_col].to_numpy()
        mask = ~np.isnan(pvals)
        adj = np.full_like(pvals, np.nan, dtype=float)
        if mask.sum() > 0:
            _, p_adj, _, _ = multipletests(pvals[mask], alpha=0.05, method="fdr_bh")
            adj[mask] = p_adj
        out[out_col] = adj
    # Validation stability: sign of ARM-HAND subject-mean difference on val split
    return out


def compute_val_stability(
    df_val: pd.DataFrame,
    feat_cols: list[str],
    train_tests: pd.DataFrame,
) -> pd.Series:
    """Return boolean per-feature: direction preserved in validation."""
    arm_v = df_val[df_val["class"] == "PainArm"].groupby("subject")[feat_cols].mean()
    hand_v = df_val[df_val["class"] == "PainHand"].groupby("subject")[feat_cols].mean()
    common = arm_v.index.intersection(hand_v.index)
    out = {}
    for feat in feat_cols:
        if feat not in arm_v.columns:
            out[feat] = False
            continue
        diffs = arm_v.loc[common, feat] - hand_v.loc[common, feat]
        diffs = diffs.dropna()
        if len(diffs) == 0:
            out[feat] = False
            continue
        val_sign = np.sign(diffs.mean())
        train_sign = np.sign(
            train_tests.set_index("feature").loc[feat, "cliff_delta"]
            if feat in train_tests["feature"].values else 0
        )
        out[feat] = bool(val_sign == train_sign and val_sign != 0)
    return pd.Series(out, name="val_direction_preserved")


# ---------------------------------------------------------------------------
# Step 4: per-subject within-subject logistic regression
# ---------------------------------------------------------------------------
def per_subject_classification(
    df_train: pd.DataFrame, top_feats: list[str]
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    ph = df_train[df_train["class"].isin(["PainArm", "PainHand"])].copy()
    ph["y"] = (ph["class"] == "PainHand").astype(int)
    rows = []
    for subj in tqdm(sorted(ph["subject"].unique()), desc="per-subject"):
        sub = ph[ph["subject"] == subj]
        X = sub[top_feats].to_numpy(dtype=float)
        y = sub["y"].to_numpy()
        # Drop columns that are all-NaN for this subject, median-impute others
        col_valid = ~np.all(np.isnan(X), axis=0)
        if col_valid.sum() == 0:
            rows.append({"subject": subj, "n_segments": len(sub), "loo_acc": np.nan})
            continue
        X = X[:, col_valid]
        col_med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), col_med, X)
        # Need at least both classes
        if len(np.unique(y)) < 2 or len(y) < 6:
            rows.append({"subject": subj, "n_segments": len(sub), "loo_acc": np.nan})
            continue
        loo = LeaveOneOut()
        correct = 0
        total = 0
        for tr, te in loo.split(X):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            correct += int(pred[0] == y[te][0])
            total += 1
        acc = correct / total if total else np.nan
        rows.append({"subject": subj, "n_segments": len(sub), "loo_acc": acc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_window_profile(df_train: pd.DataFrame) -> None:
    # For each signal, plot mean window-profile across subjects (subject-z'd)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
    for i, sig in enumerate(SIGNALS):
        mean_cols = [f"{sig}_w{j}_mean" for j in range(1, 6)]
        # subject-z per feature column
        sub = df_train[df_train["class"].isin(["PainArm", "PainHand"])][
            ["subject", "class"] + mean_cols
        ].copy()
        # subject-z
        for c in mean_cols:
            gm = sub.groupby("subject")[c].transform("mean")
            gs = sub.groupby("subject")[c].transform("std")
            sub[c] = (sub[c] - gm) / gs.replace(0, np.nan)
        # Plot mean curve per class across subjects (subject-means, then class-means)
        ax = axes[i]
        for cls, color in [("PainArm", "#dd8452"), ("PainHand", "#c44e52")]:
            sub_cls = sub[sub["class"] == cls]
            # subject-mean then class mean
            subj_m = sub_cls.groupby("subject")[mean_cols].mean()
            cls_mean = subj_m.mean(axis=0).to_numpy()
            cls_sem = subj_m.std(axis=0).to_numpy() / np.sqrt(len(subj_m))
            ax.plot(WINDOW_CENTER_TIMES, cls_mean, "-o", color=color, label=cls)
            ax.fill_between(
                WINDOW_CENTER_TIMES,
                cls_mean - cls_sem,
                cls_mean + cls_sem,
                color=color, alpha=0.2,
            )
        ax.set_title(sig)
        ax.set_xlabel("Window center (s)")
        if i == 0:
            ax.set_ylabel("Subject-z window mean")
        ax.legend(fontsize=9)
    fig.suptitle("Window-mean trajectory per class (train, subject-z)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "window_profile_per_class.png")
    plt.close(fig)


def plot_subject_hist(per_subj: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    accs = per_subj["loo_acc"].dropna()
    ax.hist(accs, bins=np.linspace(0, 1, 21), color="#4c72b0", edgecolor="k")
    ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8, label="chance")
    ax.axvline(0.7, color="#c44e52", linestyle="--", linewidth=0.8, label="0.7 cutoff")
    ax.set_xlabel("Per-subject LOO accuracy (ARM vs HAND)")
    ax.set_ylabel("Subjects")
    ax.set_title(f"Within-subject discriminability (n={len(accs)} subjects)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "subject_discriminability_hist.png")
    plt.close(fig)


def plot_top20_wilcoxon(tests: pd.DataFrame) -> None:
    t = tests.dropna(subset=["paired_W"]).copy()
    t["absW"] = t["paired_W"].abs()
    # Because Wilcoxon W is the smaller of T+/T-, "top" by signal strength = smallest p
    t = t.sort_values("paired_p").head(20)
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = [
        {"Bvp": "#4c72b0", "Eda": "#55a868", "Resp": "#dd8452", "SpO2": "#c44e52"}.get(
            s, "#888"
        )
        for s in t["signal"]
    ]
    y = np.arange(len(t))
    ax.barh(y, -np.log10(t["paired_p"]), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(t["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(paired Wilcoxon p)")
    ax.set_title("Top 20 temporal features by paired Wilcoxon p (ARM vs HAND)")
    for i, (_, r) in enumerate(t.iterrows()):
        ax.text(
            -np.log10(r["paired_p"]) + 0.02, i,
            f"q={r['paired_p_fdr']:.2g}",
            va="center", fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top20_paired_wilcoxon_bar.png")
    plt.close(fig)


def plot_cliff_vs_sign(tests: pd.DataFrame) -> None:
    d = tests.dropna(subset=["cliff_delta", "sign_consistency_frac"]).copy()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        np.abs(d["cliff_delta"]),
        d["sign_consistency_frac"],
        s=18, alpha=0.5, c="#4c72b0",
    )
    ax.axhline(0.5, color="gray", linewidth=0.5)
    ax.set_xlabel("|Cliff's delta| (subject means)")
    ax.set_ylabel("Sign consistency (fraction ARM > HAND)")
    ax.set_title("Effect size vs direction consistency per feature")
    # label top 10 by |sign - 0.5|
    d["devi"] = np.abs(d["sign_consistency_frac"] - 0.5)
    top = d.sort_values("devi", ascending=False).head(10)
    for _, r in top.iterrows():
        ax.annotate(
            r["feature"], (abs(r["cliff_delta"]), r["sign_consistency_frac"]),
            fontsize=7, alpha=0.85,
        )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cliff_delta_vs_sign_consistency.png")
    plt.close(fig)


def plot_slope_distribution(df_train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for i, sig in enumerate(SIGNALS):
        col = f"{sig}_slope_mean"
        sub = df_train[df_train["class"].isin(["PainArm", "PainHand"])][
            ["class", col]
        ].dropna()
        if len(sub) == 0:
            axes[i].set_title(sig)
            continue
        sns.violinplot(
            data=sub, x="class", y=col, ax=axes[i],
            palette={"PainArm": "#dd8452", "PainHand": "#c44e52"}, inner="box",
        )
        axes[i].set_title(sig)
        axes[i].set_xlabel("")
        if i == 0:
            axes[i].set_ylabel("slope of window-mean")
        else:
            axes[i].set_ylabel("")
    fig.suptitle("Within-segment slope of window-mean, by class")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "slope_distribution_by_class.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(
    tests: pd.DataFrame,
    per_subj: pd.DataFrame,
    consistent: pd.DataFrame,
    n_features: int,
    n_train_subjects: int,
    val_stability: pd.Series,
) -> None:
    n_fdr_paired = int((tests["paired_p_fdr"] < 0.05).sum())
    n_fdr_lmm = int((tests["mixed_lmm_p_fdr"] < 0.05).sum())
    top10 = tests.dropna(subset=["paired_p"]).sort_values("paired_p").head(10)
    sign_top = tests.dropna(subset=["sign_consistency_frac"]).copy()
    sign_top["devi"] = np.abs(sign_top["sign_consistency_frac"] - 0.5)
    sign_top = sign_top.sort_values("devi", ascending=False).head(10)
    n_disc = int((per_subj["loo_acc"] > 0.7).sum())
    n_with_acc = int(per_subj["loo_acc"].notna().sum())
    median_acc = float(per_subj["loo_acc"].median()) if n_with_acc else np.nan

    lines = []
    lines.append("# 09 Temporal dynamics: PainArm vs PainHand")
    lines.append("")
    lines.append(
        f"Per-segment 10-s signal split into 5 non-overlapping 2-s windows. "
        f"Per-window stats + trajectory summaries -> {n_features} temporal features "
        f"across {len(SIGNALS)} signals. Paired within-subject tests on "
        f"{n_train_subjects} train subjects (each subject -> one ARM score + one HAND "
        f"score per feature)."
    )
    lines.append("")
    lines.append("## FDR-significant features")
    lines.append(
        f"- Paired Wilcoxon (subject-mean ARM vs HAND), FDR q<0.05: **{n_fdr_paired}** "
        f"of {n_features}."
    )
    lines.append(
        f"- Mixed-effects LMM (segment-level, random subject intercept), FDR q<0.05: "
        f"**{n_fdr_lmm}** of {n_features}."
    )
    lines.append("")
    lines.append("## Top 10 features by smallest paired-Wilcoxon p")
    lines.append("")
    lines.append("| feature | signal | paired_p | paired_p_fdr | cliff_delta | sign_cons | lmm_coef | lmm_p | val_sign_preserved |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|:---:|")
    for _, r in top10.iterrows():
        vs = val_stability.get(r["feature"], False)
        coef = r["mixed_lmm_coef"]
        coef_str = f"{coef:.4g}" if np.isfinite(coef) else "nan"
        lmm_p = r["mixed_lmm_p"]
        lmm_p_str = f"{lmm_p:.2e}" if np.isfinite(lmm_p) else "nan"
        lines.append(
            f"| `{r['feature']}` | {r['signal']} | {r['paired_p']:.2e} | "
            f"{r['paired_p_fdr']:.2e} | {r['cliff_delta']:.3f} | "
            f"{r['sign_consistency_frac']:.3f} | "
            f"{coef_str} | {lmm_p_str} | {'yes' if vs else 'no'} |"
        )
    lines.append("")
    lines.append("## Top 10 features by sign-consistency (|p - 0.5|)")
    lines.append("")
    lines.append("| feature | signal | sign_cons | cliff_delta | paired_p | paired_p_fdr |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in sign_top.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['signal']} | "
            f"{r['sign_consistency_frac']:.3f} | {r['cliff_delta']:.3f} | "
            f"{r['paired_p']:.2e} | {r['paired_p_fdr']:.2e} |"
        )
    lines.append("")
    lines.append("## Within-subject discriminability")
    lines.append(
        f"- Subjects with evaluable LOO CV: **{n_with_acc}**."
    )
    lines.append(
        f"- Median per-subject LOO accuracy (ARM vs HAND): **{median_acc:.3f}** "
        f"(chance = 0.5)."
    )
    lines.append(
        f"- Subjects with LOO > 0.7 ('discriminable'): **{n_disc}**."
    )
    if n_disc > 0:
        disc = per_subj[per_subj["loo_acc"] > 0.7].sort_values("loo_acc", ascending=False)
        lines.append("")
        lines.append("Discriminable subjects:")
        lines.append("")
        lines.append("| subject | n_segments | LOO accuracy |")
        lines.append("|---:|---:|---:|")
        for _, r in disc.iterrows():
            lines.append(f"| {int(r['subject'])} | {int(r['n_segments'])} | {r['loo_acc']:.3f} |")
    lines.append("")
    lines.append("## Recommendation")
    if n_fdr_paired > 0 or n_disc >= 5:
        lines.append(
            "- Temporal structure carries some ARM-vs-HAND signal: "
            f"{n_fdr_paired} features survive FDR, and {n_disc} subjects "
            "are personally discriminable. Pursue per-subject modelling + "
            "temporal features as a weak-but-real signal. A global classifier "
            "will still struggle unless subject-specific calibration is allowed."
        )
    else:
        lines.append(
            "- Temporal structure does not materially improve global ARM-vs-HAND "
            f"separability: {n_fdr_paired} features at FDR, and only {n_disc} "
            "subjects are personally discriminable above 0.7. This dataset likely "
            "lacks a systematic, subject-invariant arm-vs-hand fingerprint in "
            "the 4 physiological channels at 100 Hz. Any usable separation would "
            "require either richer morphology features, subject calibration, or "
            "accepting near-chance performance."
        )
    lines.append("")
    lines.append("## Methodological notes")
    lines.append(
        "- SpO2 flatline segments (std < 1e-6) are set to NaN for all SpO2 features."
    )
    lines.append(
        "- Subject-z delta features are computed as (feature - subject's NoPain mean) "
        "using both train and validation NoPain segments per subject."
    )
    lines.append(
        "- Paired Wilcoxon operates on subject-means (one ARM score + one HAND score "
        "per subject). Mixed LMM uses all ARM+HAND segments with subject random intercepts."
    )
    lines.append(
        "- FDR (BH, alpha=0.05) applied separately for paired-Wilcoxon and LMM families."
    )
    lines.append(
        "- Validation direction-preservation is a coarse sanity check using sign of "
        "ARM-HAND difference on the validation split."
    )
    lines.append("")
    (REPORTS_DIR / "09_temporal_armhand_summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parquet_fp = TABLES_DIR / "temporal_features.parquet"
    cache_fp = TABLES_DIR / "temporal_features_allclass.parquet"

    if cache_fp.exists():
        print("[1/6] Loading cached full-class temporal feature matrix...")
        df = pd.read_parquet(cache_fp)
        feat_cols = [
            c for c in df.columns
            if c not in META_COLS and not c.endswith("__delta")
        ]
        delta_cols = [c for c in df.columns if c.endswith("__delta")]
        all_cols = feat_cols + delta_cols
        print(f"  loaded {df.shape}, base={len(feat_cols)}, delta={len(delta_cols)}")
    else:
        print("[1/6] Loading tensors and meta...")
        train_tensor, train_meta = load_split("train")
        val_tensor, val_meta = load_split("validation")
        inv = pd.read_csv(TABLES_DIR / "inventory_constant_segments.csv")
        spo2_flat = inv[inv["signal"] == "SpO2"]
        flatline_ids: set[str] = set(spo2_flat["segment_id"].tolist())
        print(f"  train {train_tensor.shape}, val {val_tensor.shape}, "
              f"spo2 flatline segments: {len(flatline_ids)}")

        print("[2/6] Building temporal feature matrix...")
        df = build_temporal_features(
            train_tensor, train_meta, val_tensor, val_meta, flatline_ids
        )
        feat_cols = [c for c in df.columns if c not in META_COLS]
        print(f"  raw temporal features: {len(feat_cols)}")

        print("[3/6] Adding subject-NoPain delta features...")
        df = add_subject_delta(df, feat_cols)
        delta_cols = [c for c in df.columns if c.endswith("__delta")]
        all_cols = feat_cols + delta_cols
        print(f"  total features (base + delta): {len(all_cols)}")
        df.to_parquet(cache_fp, index=False)

    pain_only = df[df["class"].isin(["PainArm", "PainHand"])].copy()
    save_cols = META_COLS + all_cols
    pain_only[save_cols].to_parquet(parquet_fp, index=False)
    print(f"  wrote temporal_features.parquet ({pain_only.shape})")

    print("[4/6] Running paired-Wilcoxon + mixed-LMM tests (train)...")
    train_df = df[df["split"] == "train"].copy()
    tests = run_arm_vs_hand_tests(train_df, all_cols)

    val_df = df[df["split"] == "validation"].copy()
    val_stab = compute_val_stability(val_df, all_cols, tests)
    tests["val_direction_preserved"] = tests["feature"].map(val_stab)

    tests_sorted = tests.sort_values("paired_p").reset_index(drop=True)
    keep_cols = [
        "feature", "signal", "n_subjects_paired",
        "paired_W", "paired_p", "paired_p_fdr",
        "cliff_delta", "sign_consistency_frac",
        "mixed_lmm_coef", "mixed_lmm_p", "mixed_lmm_p_fdr",
        "val_direction_preserved",
    ]
    tests_sorted[keep_cols].to_csv(
        TABLES_DIR / "temporal_armhand_tests.csv", index=False
    )
    n_fdr = int((tests["paired_p_fdr"] < 0.05).sum())
    print(f"  paired-Wilcoxon FDR<0.05: {n_fdr}")
    print(f"  LMM FDR<0.05: {int((tests['mixed_lmm_p_fdr'] < 0.05).sum())}")

    print("[5/6] Per-subject within-subject logistic regression (top-30)...")
    # Top 30 by paired_p (smallest p = strongest)
    top30 = (
        tests.dropna(subset=["paired_p"])
        .sort_values("paired_p").head(30)["feature"].tolist()
    )
    per_subj = per_subject_classification(train_df, top30)
    per_subj.to_csv(
        TABLES_DIR / "temporal_armhand_per_subject.csv", index=False
    )

    # Sign-consistency leaders
    sign_lead = tests.dropna(subset=["sign_consistency_frac"]).copy()
    sign_lead["devi"] = np.abs(sign_lead["sign_consistency_frac"] - 0.5)
    sign_lead = sign_lead.sort_values("devi", ascending=False).head(30)
    sign_lead[
        ["feature", "signal", "sign_consistency_frac", "cliff_delta",
         "paired_p", "paired_p_fdr", "n_subjects_paired"]
    ].to_csv(TABLES_DIR / "temporal_armhand_consistent.csv", index=False)

    print("[6/6] Plotting + writing report...")
    plot_window_profile(train_df)
    plot_subject_hist(per_subj)
    plot_top20_wilcoxon(tests)
    plot_cliff_vs_sign(tests)
    plot_slope_distribution(train_df)

    write_report(
        tests, per_subj, sign_lead, len(all_cols),
        train_df["subject"].nunique(), val_stab,
    )
    print("Done.")


if __name__ == "__main__":
    main()
