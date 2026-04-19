"""Tier-B local analysis for ARM vs HAND.

Three checks not covered by previous scripts:

  1. Per-sample subject-residual waveforms (pain segment minus subject's
     mean NoPain segment) for raw signal x, first derivative dx, second
     derivative ddx. Permutation/Wilcoxon paired test per sample across
     41 train subjects -> time windows where ARM differs from HAND.

  2. Aggregate features on dx and ddx (mean/std/skew/kurt/rms/zcr/range)
     per channel. Paired Wilcoxon ARM vs HAND with BH-FDR.

  3. Preprocessing sweep: LR LOSO on flattened residual waveform under
     {none, bandpass(0.05-1), bandpass(0.1-0.5), detrend, savgol smooth}
     per channel. Pick best.

Outputs
-------
- results/tables/tierB_residual_per_sample_pvals.parquet
- results/tables/tierB_derivative_features.parquet
- results/tables/tierB_derivative_tests.csv
- results/tables/tierB_filter_sweep.csv
- plots/tierB_waveform/{mean_residual_per_channel_{x,dx,ddx}.png,
                        sample_signif_heatmap.png,
                        filter_sweep_bar.png,
                        top_derivative_features.png}
- results/reports/20_tierB_waveform_residual_summary.md
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal as sps  # noqa: E402
from scipy import stats as sstats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data_loader import SFREQ, SIGNALS, load_split  # noqa: E402

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
REP = ROOT / "results" / "reports"
PLT = ROOT / "plots" / "tierB_waveform"
for d in (TAB, REP, PLT):
    d.mkdir(parents=True, exist_ok=True)

ARM_HAND = ("PainArm", "PainHand")
LABEL_MAP = {"PainArm": 0, "PainHand": 1}
warnings.filterwarnings("ignore")


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    diffs = a[:, None] - b[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / (a.size * b.size))


# ---------------------------------------------------------------------------
# Build subject-residual waveforms
# ---------------------------------------------------------------------------
def build_residuals(tensor: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """For each segment, subtract that subject's mean NoPain waveform per
    channel. Returns same-shape tensor."""
    out = tensor.copy().astype(np.float32)
    nan_mask = np.isnan(out)
    out[nan_mask] = 0.0
    for sid in meta["subject"].unique():
        rows_all = meta.index[meta["subject"] == sid].to_numpy()
        rows_nopain = meta.index[
            (meta["subject"] == sid) & (meta["class"] == "NoPain")
        ].to_numpy()
        if rows_nopain.size == 0:
            continue
        baseline = out[rows_nopain].mean(axis=0)  # (4, T)
        out[rows_all] -= baseline[None, :, :]
    out[nan_mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# Per-sample paired Wilcoxon ARM vs HAND on subject-mean residual waveform
# ---------------------------------------------------------------------------
def per_sample_paired_wilcoxon(
    residuals: np.ndarray, meta: pd.DataFrame, signal_idx: int,
    derivative: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (subject_mean_arm, subject_mean_hand, p_per_sample)
    arrays. Each per-sample p comes from paired Wilcoxon across subjects.
    """
    arm_means: list[np.ndarray] = []
    hand_means: list[np.ndarray] = []
    subjects = sorted(meta["subject"].unique())
    for sid in subjects:
        rows_arm = meta.index[
            (meta["subject"] == sid) & (meta["class"] == "PainArm")
        ].to_numpy()
        rows_hand = meta.index[
            (meta["subject"] == sid) & (meta["class"] == "PainHand")
        ].to_numpy()
        if rows_arm.size == 0 or rows_hand.size == 0:
            continue
        arm_seg = residuals[rows_arm, signal_idx, :]
        hand_seg = residuals[rows_hand, signal_idx, :]
        if derivative >= 1:
            arm_seg = np.diff(arm_seg, n=derivative, axis=1)
            hand_seg = np.diff(hand_seg, n=derivative, axis=1)
        arm_means.append(np.nanmean(arm_seg, axis=0))
        hand_means.append(np.nanmean(hand_seg, axis=0))
    A = np.stack(arm_means)   # (n_subj, T)
    H = np.stack(hand_means)
    T = A.shape[1]
    pvals = np.full(T, np.nan)
    for t in range(T):
        a, h = A[:, t], H[:, t]
        m = ~(np.isnan(a) | np.isnan(h))
        if m.sum() < 5 or np.allclose(a[m], h[m]):
            continue
        try:
            _, p = sstats.wilcoxon(a[m], h[m], zero_method="wilcox",
                                   alternative="two-sided", mode="auto")
            pvals[t] = float(p)
        except Exception:
            pass
    return A, H, pvals


# ---------------------------------------------------------------------------
# Aggregate-stat features on dx, ddx
# ---------------------------------------------------------------------------
def agg_derivative_features(
    residuals: np.ndarray, meta: pd.DataFrame
) -> pd.DataFrame:
    """For each (segment, signal) compute aggregate stats on dx and ddx."""
    feats: list[dict] = []
    for i in range(len(meta)):
        row = {
            "subject": int(meta.iloc[i]["subject"]),
            "class": meta.iloc[i]["class"],
            "segment_id": meta.iloc[i]["segment_id"],
            "split": meta.iloc[i]["split"],
        }
        for s_i, sig in enumerate(SIGNALS):
            x = residuals[i, s_i]
            x = x[~np.isnan(x)]
            if x.size < 4:
                continue
            for order_name, y in (("d1", np.diff(x, n=1)),
                                  ("d2", np.diff(x, n=2))):
                if y.size < 3:
                    continue
                ystd = float(np.std(y))
                row[f"{sig}_{order_name}_mean"] = float(np.mean(y))
                row[f"{sig}_{order_name}_std"] = ystd
                row[f"{sig}_{order_name}_rms"] = float(np.sqrt(np.mean(y * y)))
                row[f"{sig}_{order_name}_range"] = float(y.max() - y.min())
                row[f"{sig}_{order_name}_skew"] = (
                    float(sstats.skew(y)) if ystd > 0 else 0.0
                )
                row[f"{sig}_{order_name}_kurt"] = (
                    float(sstats.kurtosis(y)) if ystd > 0 else 0.0
                )
                # zero-crossing rate around mean
                if y.size > 1:
                    ym = y - np.mean(y)
                    row[f"{sig}_{order_name}_zcr"] = float(
                        np.mean(np.diff(np.signbit(ym).astype(int)) != 0)
                    )
                else:
                    row[f"{sig}_{order_name}_zcr"] = 0.0
        feats.append(row)
    return pd.DataFrame(feats)


def paired_wilcoxon_subj_mean(
    df: pd.DataFrame, feat_cols: list[str]
) -> pd.DataFrame:
    train = df[df["split"] == "train"]
    arm = train[train["class"] == "PainArm"].groupby("subject")[feat_cols].mean()
    hand = train[train["class"] == "PainHand"].groupby("subject")[feat_cols].mean()
    common = arm.index.intersection(hand.index)
    arm, hand = arm.loc[common], hand.loc[common]
    rows = []
    for f in feat_cols:
        a, h = arm[f].to_numpy(), hand[f].to_numpy()
        m = ~(np.isnan(a) | np.isnan(h))
        if m.sum() < 5 or np.allclose(a[m], h[m]):
            rows.append({"feature": f, "n": int(m.sum()),
                         "W": np.nan, "p": np.nan,
                         "cliff": np.nan, "sign_arm_gt_hand": np.nan,
                         "mean_arm": float(np.nanmean(a[m])) if m.any() else np.nan,
                         "mean_hand": float(np.nanmean(h[m])) if m.any() else np.nan})
            continue
        try:
            res = sstats.wilcoxon(a[m], h[m], zero_method="wilcox",
                                  alternative="two-sided", mode="auto")
            W, p = float(res.statistic), float(res.pvalue)
        except Exception:
            W, p = np.nan, np.nan
        rows.append({
            "feature": f, "n": int(m.sum()), "W": W, "p": p,
            "cliff": cliffs_delta(a[m], h[m]),
            "sign_arm_gt_hand": float(np.mean(a[m] > h[m])),
            "mean_arm": float(np.mean(a[m])),
            "mean_hand": float(np.mean(h[m])),
        })
    out = pd.DataFrame(rows)
    if out["p"].notna().any():
        idx = out.index[out["p"].notna()]
        _, p_corr, _, _ = multipletests(out.loc[idx, "p"].values,
                                        method="fdr_bh")
        out.loc[idx, "p_fdr"] = p_corr
    return out.sort_values("p", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Filter sweep with flattened-waveform LR
# ---------------------------------------------------------------------------
def apply_filter(x: np.ndarray, kind: str, fs: int = SFREQ) -> np.ndarray:
    if kind == "none":
        return x
    if kind == "bp_0.05_1":
        b, a = sps.butter(3, [0.05, 1.0], btype="band", fs=fs)
        return sps.filtfilt(b, a, x, axis=-1)
    if kind == "bp_0.1_0.5":
        b, a = sps.butter(3, [0.1, 0.5], btype="band", fs=fs)
        return sps.filtfilt(b, a, x, axis=-1)
    if kind == "detrend":
        return sps.detrend(x, axis=-1, type="linear")
    if kind == "savgol_25":
        win = 25 if x.shape[-1] >= 25 else max(5, x.shape[-1] // 4 * 2 + 1)
        if win % 2 == 0:
            win += 1
        return sps.savgol_filter(x, win, 3, axis=-1)
    raise ValueError(kind)


def filter_sweep(
    residuals: np.ndarray, meta: pd.DataFrame, downsample: int = 4,
) -> pd.DataFrame:
    train_mask = (meta["class"].isin(ARM_HAND) & (meta["split"] == "train")).to_numpy()
    R = residuals[train_mask]
    sub_meta = meta[train_mask].reset_index(drop=True)
    y = sub_meta["class"].map(LABEL_MAP).to_numpy()
    subj = sub_meta["subject"].to_numpy()
    R = np.nan_to_num(R, nan=0.0)

    rows = []
    filters = ["none", "bp_0.05_1", "bp_0.1_0.5", "detrend", "savgol_25"]
    for kind in filters:
        for s_i, sig in enumerate(SIGNALS):
            try:
                X = apply_filter(R[:, s_i, :], kind)
            except Exception as e:
                print(f"  filter {kind} sig {sig} failed: {e}")
                continue
            X = X[:, ::downsample]  # 1000 -> 250 samples
            X = X.astype(np.float32)
            logo = LeaveOneGroupOut()
            f1s = []
            for tr, te in logo.split(X, y, groups=subj):
                sc = StandardScaler().fit(X[tr])
                mdl = LogisticRegression(
                    penalty="l2", C=1.0, class_weight="balanced",
                    max_iter=4000, solver="lbfgs", random_state=SEED, n_jobs=1,
                )
                mdl.fit(sc.transform(X[tr]), y[tr])
                yhat = mdl.predict(sc.transform(X[te]))
                f1s.append(f1_score(y[te], yhat, average="macro",
                                    zero_division=0))
            rows.append({
                "filter": kind, "signal": sig,
                "loso_macro_f1_mean": float(np.mean(f1s)),
                "loso_macro_f1_std": float(np.std(f1s)),
                "n_samples_per_seg": X.shape[1],
            })
            print(f"  {kind:<12} {sig:<6} F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_mean_residual(
    A_dict: dict, H_dict: dict, P_dict: dict, deriv_label: str
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=False)
    for ax, sig in zip(axes, SIGNALS):
        A = A_dict[sig]; H = H_dict[sig]; P = P_dict[sig]
        T = A.shape[1]
        t = np.arange(T) / SFREQ
        a_mean = np.nanmean(A, axis=0); h_mean = np.nanmean(H, axis=0)
        a_sem = np.nanstd(A, axis=0) / np.sqrt(np.maximum(np.sum(~np.isnan(A), axis=0), 1))
        h_sem = np.nanstd(H, axis=0) / np.sqrt(np.maximum(np.sum(~np.isnan(H), axis=0), 1))
        ax.plot(t, a_mean, color="#dd8452", label="ARM", linewidth=1.4)
        ax.fill_between(t, a_mean - a_sem, a_mean + a_sem,
                        color="#dd8452", alpha=0.2)
        ax.plot(t, h_mean, color="#c44e52", label="HAND", linewidth=1.4)
        ax.fill_between(t, h_mean - h_sem, h_mean + h_sem,
                        color="#c44e52", alpha=0.2)
        # mark significant samples (raw p < 0.05)
        sig_mask = P < 0.05
        if sig_mask.any():
            ax.scatter(t[sig_mask],
                       np.full(sig_mask.sum(), ax.get_ylim()[0]),
                       color="black", s=2, label="p<0.05")
        ax.set_title(f"{sig} ({deriv_label})", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        if sig == SIGNALS[0]:
            ax.set_ylabel("subject-residual mean")
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Subject-residual mean waveform — ARM vs HAND ({deriv_label})")
    fig.tight_layout()
    fig.savefig(PLT / f"mean_residual_per_channel_{deriv_label}.png", dpi=130)
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def plot_signif_heatmap(P_dict: dict) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 4), sharex=True)
    deriv_order = ["x", "dx", "ddx"]
    for ax, label in zip(axes, deriv_order):
        rows = []
        for sig in SIGNALS:
            P = P_dict[(label, sig)]
            row = -np.log10(np.where(np.isnan(P), 1.0, P))
            rows.append(row)
        M = np.stack(rows)
        im = ax.imshow(M, aspect="auto", cmap="magma",
                       vmin=0, vmax=3, interpolation="nearest",
                       extent=[0, M.shape[1] / SFREQ, 0, len(SIGNALS)])
        ax.set_yticks(np.arange(len(SIGNALS)) + 0.5)
        ax.set_yticklabels(list(SIGNALS), fontsize=8)
        ax.set_ylabel(label)
        if label == deriv_order[-1]:
            ax.set_xlabel("time (s)")
        plt.colorbar(im, ax=ax, label="-log10(p)", shrink=0.8)
    fig.suptitle("Per-sample paired Wilcoxon ARM vs HAND (-log10 p)")
    fig.tight_layout()
    fig.savefig(PLT / "sample_signif_heatmap.png", dpi=130)
    plt.close(fig)


def plot_filter_sweep(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot = df.pivot(index="filter", columns="signal", values="loso_macro_f1_mean")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax,
                vmin=0.4, vmax=0.65, cbar_kws={"label": "LOSO macro-F1"})
    ax.set_title("Filter sweep — LR on flattened residual waveform per channel")
    fig.tight_layout()
    fig.savefig(PLT / "filter_sweep_bar.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[load] train + val tensors ...")
    train_t, train_m = load_split("train")
    val_t, val_m = load_split("validation")
    train_m = train_m.assign(split="train")
    val_m = val_m.assign(split="validation")
    full_t = np.concatenate([train_t, val_t], axis=0)
    full_m = pd.concat([train_m, val_m], ignore_index=True)
    print(f"[load] full tensor {full_t.shape}, {len(full_m)} segments")

    print("[1] building subject-residual waveforms ...")
    R = build_residuals(full_t, full_m)

    # --- Per-sample paired Wilcoxon for raw / dx / ddx ---
    print("[2] per-sample paired Wilcoxon ARM vs HAND (train) ...")
    train_only_mask = (full_m["split"] == "train").to_numpy()
    R_train = R[train_only_mask]
    M_train = full_m[train_only_mask].reset_index(drop=True)

    A_dict_x: dict[str, np.ndarray] = {}; H_dict_x: dict[str, np.ndarray] = {}; P_dict_x: dict[str, np.ndarray] = {}
    A_dict_dx: dict[str, np.ndarray] = {}; H_dict_dx: dict[str, np.ndarray] = {}; P_dict_dx: dict[str, np.ndarray] = {}
    A_dict_ddx: dict[str, np.ndarray] = {}; H_dict_ddx: dict[str, np.ndarray] = {}; P_dict_ddx: dict[str, np.ndarray] = {}
    for s_i, sig in enumerate(tqdm(SIGNALS, desc="per-sample stats")):
        A_dict_x[sig], H_dict_x[sig], P_dict_x[sig] = per_sample_paired_wilcoxon(R_train, M_train, s_i, derivative=0)
        A_dict_dx[sig], H_dict_dx[sig], P_dict_dx[sig] = per_sample_paired_wilcoxon(R_train, M_train, s_i, derivative=1)
        A_dict_ddx[sig], H_dict_ddx[sig], P_dict_ddx[sig] = per_sample_paired_wilcoxon(R_train, M_train, s_i, derivative=2)

    # Save per-sample p-values (long form)
    rows = []
    for label, P_dict in (("x", P_dict_x), ("dx", P_dict_dx), ("ddx", P_dict_ddx)):
        for sig in SIGNALS:
            P = P_dict[sig]
            for t_idx, p in enumerate(P):
                rows.append({"derivative": label, "signal": sig,
                             "sample_idx": t_idx, "time_s": t_idx / SFREQ,
                             "p": p})
    psdf = pd.DataFrame(rows)
    # FDR per (derivative, signal) family
    psdf["p_fdr"] = np.nan
    for (lab, sig), grp in psdf.groupby(["derivative", "signal"]):
        idx = grp.index[grp["p"].notna()]
        if idx.size:
            _, p_corr, _, _ = multipletests(grp.loc[idx, "p"].values,
                                            method="fdr_bh")
            psdf.loc[idx, "p_fdr"] = p_corr
    psdf.to_parquet(TAB / "tierB_residual_per_sample_pvals.parquet", index=False)

    plot_mean_residual(A_dict_x, H_dict_x, P_dict_x, "x")
    plot_mean_residual(A_dict_dx, H_dict_dx, P_dict_dx, "dx")
    plot_mean_residual(A_dict_ddx, H_dict_ddx, P_dict_ddx, "ddx")
    plot_signif_heatmap({
        ("x", s): P_dict_x[s] for s in SIGNALS
    } | {
        ("dx", s): P_dict_dx[s] for s in SIGNALS
    } | {
        ("ddx", s): P_dict_ddx[s] for s in SIGNALS
    })

    # --- Aggregate dx, ddx features ---
    print("[3] aggregate-stat features on dx, ddx ...")
    df_d = agg_derivative_features(R, full_m)
    df_d.to_parquet(TAB / "tierB_derivative_features.parquet", index=False)
    feat_cols = [c for c in df_d.columns
                 if c not in ("subject", "class", "segment_id", "split")]
    tests = paired_wilcoxon_subj_mean(df_d, feat_cols)
    tests.to_csv(TAB / "tierB_derivative_tests.csv", index=False)
    n_fdr05 = int((tests["p_fdr"] < 0.05).sum()) if "p_fdr" in tests else 0
    n_p01 = int((tests["p"] < 0.01).sum())
    print(f"  derivative features: {n_fdr05} FDR<0.05, {n_p01} raw p<0.01")

    # --- Filter sweep ---
    print("[4] filter sweep ...")
    sweep = filter_sweep(R, full_m)
    sweep.to_csv(TAB / "tierB_filter_sweep.csv", index=False)
    plot_filter_sweep(sweep)
    best = sweep.sort_values("loso_macro_f1_mean", ascending=False).head(5)
    print("  best 5 (filter, signal):")
    print(best.to_string(index=False))

    # --- Report ---
    n_sig_per = (psdf
                 .assign(sig=psdf["p"] < 0.05)
                 .groupby(["derivative", "signal"])["sig"]
                 .agg(["sum", "count"])
                 .reset_index())
    n_fdr_per = (psdf
                 .assign(fdr=psdf["p_fdr"] < 0.05)
                 .groupby(["derivative", "signal"])["fdr"]
                 .agg(["sum", "count"])
                 .reset_index())

    lines = ["# 20 — Tier-B waveform residual + filter sweep\n"]
    lines.append("## Q1 — per-sample paired Wilcoxon (subject-residual)\n")
    lines.append("Counts of samples with p<0.05 / FDR<0.05 across the 1000-sample window:\n")
    lines.append("| derivative | signal | n_p<0.05 | n_FDR<0.05 |")
    lines.append("|---|---|---|---|")
    for _, r in n_sig_per.iterrows():
        fdr = n_fdr_per[(n_fdr_per["derivative"] == r["derivative"])
                        & (n_fdr_per["signal"] == r["signal"])]["sum"].iloc[0]
        lines.append(f"| {r['derivative']} | {r['signal']} | {int(r['sum'])} | {int(fdr)} |")
    lines.append("")
    lines.append("## Q2 — aggregate dx/ddx features (paired Wilcoxon FDR)\n")
    lines.append(f"- {n_fdr05}/{len(tests)} features survive BH-FDR<0.05")
    lines.append(f"- {n_p01}/{len(tests)} have raw p<0.01")
    lines.append("\nTop 10 by smallest p:\n")
    lines.append(tests.head(10)[["feature", "p", "p_fdr",
                                 "cliff", "sign_arm_gt_hand",
                                 "mean_arm", "mean_hand"]]
                 .to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Q3 — filter sweep (per-channel LR LOSO on flattened residual)\n")
    lines.append("Chance = 0.50.\n")
    lines.append(sweep.sort_values("loso_macro_f1_mean", ascending=False)
                 .to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Outputs\n")
    lines.append("- `results/tables/tierB_residual_per_sample_pvals.parquet`")
    lines.append("- `results/tables/tierB_derivative_features.parquet`")
    lines.append("- `results/tables/tierB_derivative_tests.csv`")
    lines.append("- `results/tables/tierB_filter_sweep.csv`")
    lines.append("- `plots/tierB_waveform/{mean_residual_per_channel_*,sample_signif_heatmap,filter_sweep_bar}.png`")
    (REP / "20_tierB_waveform_residual_summary.md").write_text("\n".join(lines))
    print(f"\n[save] report -> {REP / '20_tierB_waveform_residual_summary.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
