"""
Stage-2 sub-window vote ensemble for arm-vs-hand.

Idea: each 10.22s segment is split into N sub-windows. Compact RESP features
extracted per sub-window. Each sub-window is classified independently. Segment
prob = mean of its sub-window probs -> noise averaging, ~sqrt(N) variance
reduction, targets weak-amplitude subjects.

Pipeline:
  1. Load raw tensor (N_seg, 4, 1022) via data_loader.load_split.
  2. Slice into n_sub equal sub-windows (non-overlap).
  3. Per (segment, sub-window) compute RESP stats (mean, std, slope, range,
     peak-to-peak, rms, median, mad, pos_frac, autocorr1).
  4. Concat train sub-window rows -> logreg with isotonic calibration.
  5. LOSO over train subjects. Aggregate sub-window probs by segment_id (mean).
  6. Decode arm-vs-hand via exact-12 per subject.

Compares against sub-window CONCAT (single row per segment, n_sub*feats wide)
and a single-window RESP-compact baseline (n_sub=1).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from src.data_loader import SIGNALS, load_split
from src.final_pipeline import (
    ARM_HAND,
    armhand_binary,
    fit_binary_calibrator,
    metrics_binary,
)

SEED = 42
RESP_IDX = SIGNALS.index("Resp")
BVP_IDX = SIGNALS.index("Bvp")
EDA_IDX = SIGNALS.index("Eda")


def _safe(fn, x, default=0.0):
    x = x[np.isfinite(x)]
    if x.size < 2:
        return default
    try:
        return float(fn(x))
    except Exception:
        return default


def subwindow_features(x: np.ndarray) -> dict[str, float]:
    """Compact stats for a 1D sub-window signal."""
    x = x[np.isfinite(x)]
    if x.size < 3:
        return {k: 0.0 for k in ("mean", "std", "slope", "range", "ptp", "rms",
                                   "median", "mad", "pos_frac", "ac1", "skew")}
    mean = float(np.mean(x))
    std = float(np.std(x))
    t = np.arange(x.size, dtype=np.float64)
    try:
        slope = float(np.polyfit(t, x.astype(np.float64), 1)[0])
    except Exception:
        slope = 0.0
    rng = float(x.max() - x.min())
    ptp = rng
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    pos_frac = float(np.mean(x > mean))
    if x.size > 1:
        x_c = x - mean
        denom = float(np.sum(x_c * x_c))
        ac1 = float(np.sum(x_c[:-1] * x_c[1:]) / denom) if denom > 0 else 0.0
    else:
        ac1 = 0.0
    if std > 1e-8:
        skew = float(np.mean(((x - mean) / std) ** 3))
    else:
        skew = 0.0
    return {"mean": mean, "std": std, "slope": slope, "range": rng, "ptp": ptp,
            "rms": rms, "median": med, "mad": mad, "pos_frac": pos_frac,
            "ac1": ac1, "skew": skew}


def build_subwindow_table(tensor: np.ndarray, meta: pd.DataFrame, n_sub: int,
                            signals: list[str], split: str) -> pd.DataFrame:
    total_len = tensor.shape[-1]
    sub_len = total_len // n_sub
    sig_indices = [SIGNALS.index(s) for s in signals]

    rows = []
    for i in range(len(meta)):
        m = meta.iloc[i]
        base = {
            "split": split,
            "subject": int(m["subject"]),
            "class": m["class"],
            "segment_id": m["segment_id"],
            "segment_idx": int(m["segment_idx"]),
        }
        for w in range(n_sub):
            a, b = w * sub_len, (w + 1) * sub_len
            feats = {}
            for sig, s_i in zip(signals, sig_indices):
                sub = tensor[i, s_i, a:b]
                for k, v in subwindow_features(sub).items():
                    feats[f"{sig}_sw_{k}"] = v
            rows.append({**base, "subwindow": w, **feats})
    return pd.DataFrame(rows)


def exact12_arm_predictions(df_seg: pd.DataFrame, arm_scores: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df_seg), dtype=int)
    for subject in sorted(df_seg["subject"].unique()):
        mask = (df_seg["subject"] == subject).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-arm_scores[mask])]
        arm_idx = set(order[:12].tolist())
        pred[mask] = [0 if i in arm_idx else 1 for i in idx]
    return pred


def subject_z_norm(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    mu = out.groupby("subject")[feat_cols].transform("mean")
    sd = out.groupby("subject")[feat_cols].transform("std", ddof=0)
    sd = sd.where(sd > 0, 1.0)
    out[feat_cols] = ((out[feat_cols] - mu) / sd).fillna(0.0).astype(np.float32)
    return out


def fit_logreg_proba(X_tr: np.ndarray, X_te: np.ndarray, y_tr: np.ndarray, C: float = 1.0) -> np.ndarray:
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    mdl = LogisticRegression(C=C, class_weight="balanced", max_iter=4000,
                              solver="lbfgs", random_state=SEED)
    mdl.fit(X_tr, y_tr)
    return mdl.predict_proba(X_te).astype(np.float32)


def run_vote(sw_df: pd.DataFrame, feat_cols: list[str], calibration: str) -> dict:
    """Classify each sub-window, aggregate probs per segment by mean."""
    sw_norm = subject_z_norm(sw_df, feat_cols)
    train_sw = sw_norm[sw_norm["split"] == "train"].reset_index(drop=True)
    val_sw = sw_norm[sw_norm["split"] == "validation"].reset_index(drop=True)

    train_pain_sw = train_sw[train_sw["class"].isin(ARM_HAND)].reset_index(drop=True)
    val_pain_sw = val_sw[val_sw["class"].isin(ARM_HAND)].reset_index(drop=True)

    # LOSO on train sub-window rows
    train_sw_probs = np.zeros(len(train_pain_sw), dtype=np.float32)
    for subj in sorted(train_pain_sw["subject"].unique()):
        tr_mask = (train_pain_sw["subject"] != subj).to_numpy()
        te_mask = ~tr_mask
        X_tr = train_pain_sw.loc[tr_mask, feat_cols].to_numpy(dtype=np.float32)
        X_te = train_pain_sw.loc[te_mask, feat_cols].to_numpy(dtype=np.float32)
        y_tr = armhand_binary(train_pain_sw.loc[tr_mask, "class"])
        p = fit_logreg_proba(X_tr, X_te, y_tr)[:, 0]
        train_sw_probs[te_mask] = p

    # Full train -> val
    X_tr = train_pain_sw[feat_cols].to_numpy(dtype=np.float32)
    X_val = val_pain_sw[feat_cols].to_numpy(dtype=np.float32)
    y_tr = armhand_binary(train_pain_sw["class"])
    val_sw_probs = fit_logreg_proba(X_tr, X_val, y_tr)[:, 0]

    # Aggregate by segment_id -> mean sub-window prob
    train_seg = train_pain_sw.groupby("segment_id", sort=False).agg(
        subject=("subject", "first"),
        cls=("class", "first"),
    ).reset_index()
    train_seg["arm_prob_raw"] = (
        pd.Series(train_sw_probs, index=train_pain_sw["segment_id"].values)
        .groupby(level=0).mean().reindex(train_seg["segment_id"]).values
    )

    val_seg = val_pain_sw.groupby("segment_id", sort=False).agg(
        subject=("subject", "first"),
        cls=("class", "first"),
    ).reset_index()
    val_seg["arm_prob_raw"] = (
        pd.Series(val_sw_probs, index=val_pain_sw["segment_id"].values)
        .groupby(level=0).mean().reindex(val_seg["segment_id"]).values
    )

    # Calibrate on train
    y_train_arm = (train_seg["cls"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator(calibration, train_seg["arm_prob_raw"].to_numpy(), y_train_arm)
    train_seg["arm_prob"] = cal(train_seg["arm_prob_raw"].to_numpy())
    val_seg["arm_prob"] = cal(val_seg["arm_prob_raw"].to_numpy())

    # Decode + metrics
    results = {}
    per_subj = []
    for name, df_seg in (("loso", train_seg), ("val", val_seg)):
        df_seg = df_seg.rename(columns={"cls": "class"})
        y_true = (df_seg["class"] == "PainHand").astype(int).to_numpy()
        y_pred = exact12_arm_predictions(df_seg, df_seg["arm_prob"].to_numpy())
        m = metrics_binary(y_true, y_pred)
        results[f"{name}_macro_f1"] = m["macro_f1"]
        sub_f1s = []
        for subj in sorted(df_seg["subject"].unique()):
            mask = (df_seg["subject"] == subj).to_numpy()
            mm = metrics_binary(y_true[mask], y_pred[mask])
            sub_f1s.append(mm["macro_f1"])
            per_subj.append({"split": name, "subject": int(subj), "macro_f1": mm["macro_f1"]})
        results[f"{name}_per_subject_std"] = float(np.std(sub_f1s))
        results[f"{name}_per_subject_min"] = float(np.min(sub_f1s))
    return {"summary": results, "per_subject": pd.DataFrame(per_subj)}


def run_concat(sw_df: pd.DataFrame, feat_cols: list[str], n_sub: int, calibration: str) -> dict:
    """Reshape so each segment is one row with n_sub*len(feat_cols) features."""
    pieces = []
    for w in range(n_sub):
        sub = sw_df[sw_df["subwindow"] == w][["segment_id"] + feat_cols].copy()
        sub = sub.rename(columns={c: f"{c}_w{w}" for c in feat_cols})
        pieces.append(sub.set_index("segment_id"))
    feat_wide = pd.concat(pieces, axis=1).reset_index()
    meta_seg = sw_df[sw_df["subwindow"] == 0][["segment_id", "split", "subject", "class"]].copy()
    wide = meta_seg.merge(feat_wide, on="segment_id", how="left")
    new_feats = [c for c in wide.columns if c not in ("segment_id", "split", "subject", "class")]
    wide = subject_z_norm(wide, new_feats)

    train = wide[wide["split"] == "train"].reset_index(drop=True)
    val = wide[wide["split"] == "validation"].reset_index(drop=True)
    tp = train[train["class"].isin(ARM_HAND)].reset_index(drop=True)
    vp = val[val["class"].isin(ARM_HAND)].reset_index(drop=True)

    train_probs = np.zeros(len(tp), dtype=np.float32)
    for subj in sorted(tp["subject"].unique()):
        tr_mask = (tp["subject"] != subj).to_numpy()
        te_mask = ~tr_mask
        X_tr = tp.loc[tr_mask, new_feats].to_numpy(dtype=np.float32)
        X_te = tp.loc[te_mask, new_feats].to_numpy(dtype=np.float32)
        y_tr = armhand_binary(tp.loc[tr_mask, "class"])
        train_probs[te_mask] = fit_logreg_proba(X_tr, X_te, y_tr)[:, 0]
    X_tr = tp[new_feats].to_numpy(dtype=np.float32)
    X_val = vp[new_feats].to_numpy(dtype=np.float32)
    y_tr = armhand_binary(tp["class"])
    val_probs = fit_logreg_proba(X_tr, X_val, y_tr)[:, 0]

    y_tr_arm = (tp["class"] == "PainArm").astype(int).to_numpy()
    cal = fit_binary_calibrator(calibration, train_probs, y_tr_arm)
    tp_arm = cal(train_probs)
    vp_arm = cal(val_probs)

    results = {}
    per_subj = []
    for name, df_seg, arm in (("loso", tp, tp_arm), ("val", vp, vp_arm)):
        y_true = (df_seg["class"] == "PainHand").astype(int).to_numpy()
        y_pred = exact12_arm_predictions(df_seg, arm)
        m = metrics_binary(y_true, y_pred)
        results[f"{name}_macro_f1"] = m["macro_f1"]
        sub_f1s = []
        for subj in sorted(df_seg["subject"].unique()):
            mask = (df_seg["subject"] == subj).to_numpy()
            mm = metrics_binary(y_true[mask], y_pred[mask])
            sub_f1s.append(mm["macro_f1"])
            per_subj.append({"split": name, "subject": int(subj), "macro_f1": mm["macro_f1"]})
        results[f"{name}_per_subject_std"] = float(np.std(sub_f1s))
        results[f"{name}_per_subject_min"] = float(np.min(sub_f1s))
    return {"summary": results, "per_subject": pd.DataFrame(per_subj)}


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] building sub-window tables ...")
    tables = []
    for split in ("train", "validation"):
        tensor, meta = load_split(split, n_samples=1022)
        print(f"  {split}: tensor {tensor.shape}, segments {len(meta)}")
        sw = build_subwindow_table(tensor, meta, args.n_sub, args.signals, split)
        tables.append(sw)
    sw_df = pd.concat(tables, ignore_index=True)
    feat_cols = [c for c in sw_df.columns if "_sw_" in c]
    print(f"[features] {len(feat_cols)} sub-window feats per row, "
          f"{len(sw_df)} rows ({args.n_sub} sub-windows x {len(sw_df) // args.n_sub} segs)")

    hard10 = set([55, 46, 56, 15, 45, 25, 22, 11, 4, 63])

    rows = []
    per_subj_frames = []

    # Hard-10 LOSO summary per run
    def hard_summary(per_subj: pd.DataFrame) -> float:
        m = per_subj[(per_subj.split == "loso") & (per_subj.subject.isin(hard10))]
        return float(m["macro_f1"].mean()) if len(m) else float("nan")

    if args.mode in ("vote", "both"):
        print(f"\n[run] VOTE  n_sub={args.n_sub}")
        res = run_vote(sw_df, feat_cols, args.calibration)
        s = res["summary"]
        h = hard_summary(res["per_subject"])
        print(f"    loso={s['loso_macro_f1']:.4f} val={s['val_macro_f1']:.4f} "
              f"loso_std={s['loso_per_subject_std']:.3f} loso_min={s['loso_per_subject_min']:.3f} "
              f"hard10={h:.4f}")
        rows.append({"variant": "vote", "n_sub": args.n_sub, **s, "hard10_loso_macro_f1": h})
        pf = res["per_subject"].copy()
        pf["variant"] = "vote"
        per_subj_frames.append(pf)

    if args.mode in ("concat", "both"):
        print(f"\n[run] CONCAT n_sub={args.n_sub}")
        res = run_concat(sw_df, feat_cols, args.n_sub, args.calibration)
        s = res["summary"]
        h = hard_summary(res["per_subject"])
        print(f"    loso={s['loso_macro_f1']:.4f} val={s['val_macro_f1']:.4f} "
              f"loso_std={s['loso_per_subject_std']:.3f} loso_min={s['loso_per_subject_min']:.3f} "
              f"hard10={h:.4f}")
        rows.append({"variant": "concat", "n_sub": args.n_sub, **s, "hard10_loso_macro_f1": h})
        pf = res["per_subject"].copy()
        pf["variant"] = "concat"
        per_subj_frames.append(pf)

    summary = pd.DataFrame(rows)
    per_subj_all = pd.concat(per_subj_frames, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_subj_all.to_csv(out_dir / "per_subject.csv", index=False)

    report = ["# Stage 2 Sub-Window Vote / Concat", "",
              f"- n_sub: {args.n_sub}  signals: {args.signals}  calibration: {args.calibration}",
              f"- hard-10 set (from combined_best): {sorted(hard10)}", "",
              summary.to_markdown(index=False, floatfmt=".4f")]
    (out_dir / "report.md").write_text("\n".join(report))
    print(f"\n[done] wrote {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-sub", type=int, default=5, help="sub-windows per 10.22s segment")
    p.add_argument("--signals", nargs="+", default=["Resp"], choices=list(SIGNALS))
    p.add_argument("--calibration", default="isotonic", choices=["none", "sigmoid", "isotonic"])
    p.add_argument("--mode", choices=["vote", "concat", "both"], default="both")
    p.add_argument("--output-dir", default="results/final/stage2_subwindow")
    main(p.parse_args())
