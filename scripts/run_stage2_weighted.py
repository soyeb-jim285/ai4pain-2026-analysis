"""
Stage-2 sample-weighted training to boost weak-responder subjects.

Hypothesis: loss weighted by 1/reactivity forces model to learn arm-vs-hand
pattern in low-amplitude subjects, improving hard-subject F1 at small cost
to easy subjects.

Weighting modes:
  uniform             : baseline (w=1)
  inv_reactivity      : w_i = clip(median_reactivity / reactivity_subj, 0.5, 4)
  boost_hard          : w_i = 2.0 for bottom-k reactivity subjects, else 1.0
  equal_subject       : w_i = 1 / n_rows_subj (every subject contributes equally)
  squared_inv         : w_i = clip((median_r / r_subj)**2, 0.25, 8)

Pipeline mirrors best-stage2 config: subject_z + resp_all + logreg + robust
scaler + isotonic cal. Adds sample_weight to logreg.fit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.final_pipeline import (
    ARM_HAND,
    armhand_binary,
    fit_binary_calibrator,
    load_clean_features,
    metrics_binary,
    stage2_feature_sets,
)

SEED = 42


def subject_z_norm(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    mu = out.groupby("subject")[feat_cols].transform("mean")
    sd = out.groupby("subject")[feat_cols].transform("std", ddof=0)
    sd = sd.where(sd > 0, 1.0)
    out[feat_cols] = ((out[feat_cols] - mu) / sd).fillna(0.0).astype(np.float32)
    return out


def compute_reactivity(df: pd.DataFrame, feat_cols: list[str], ref_subjects: list[int]) -> pd.Series:
    """Per-subject L2 norm of mean(Pain) - mean(NoPain) in globally z-scored space."""
    ref = df[df["subject"].isin(ref_subjects)]
    mu = ref[feat_cols].mean(axis=0)
    sd = ref[feat_cols].std(axis=0, ddof=0).replace(0.0, 1.0)
    z = (df[feat_cols] - mu) / sd
    z = z.fillna(0.0)
    tmp = pd.concat([df[["subject", "class"]], z], axis=1)
    rows = {}
    for subj, sub in tmp.groupby("subject"):
        p = sub[sub["class"].isin(ARM_HAND)][feat_cols].mean(axis=0)
        n = sub[sub["class"] == "NoPain"][feat_cols].mean(axis=0)
        if p.isna().any() or n.isna().any():
            rows[int(subj)] = np.nan
            continue
        rows[int(subj)] = float(np.linalg.norm((p - n).to_numpy()))
    return pd.Series(rows)


def build_weights(subjects: np.ndarray, reactivity: pd.Series, mode: str, k_hard: int = 10) -> np.ndarray:
    uniq = sorted(np.unique(subjects).tolist())
    r = reactivity.reindex(uniq)
    med = float(r.median())

    if mode == "uniform":
        w_by_subj = {s: 1.0 for s in uniq}
    elif mode == "inv_reactivity":
        w_by_subj = {s: float(np.clip(med / max(r.loc[s], 1e-6), 0.5, 4.0)) for s in uniq}
    elif mode == "squared_inv":
        w_by_subj = {s: float(np.clip((med / max(r.loc[s], 1e-6)) ** 2, 0.25, 8.0)) for s in uniq}
    elif mode == "boost_hard":
        order = r.sort_values().index.tolist()
        hard = set(order[:k_hard])
        w_by_subj = {s: 2.0 if s in hard else 1.0 for s in uniq}
    elif mode == "equal_subject":
        counts = pd.Series(subjects).value_counts()
        w_by_subj = {s: 1.0 / float(counts.get(s, 1)) for s in uniq}
        m = np.mean(list(w_by_subj.values()))
        w_by_subj = {s: v / m for s, v in w_by_subj.items()}
    else:
        raise ValueError(mode)
    return np.array([w_by_subj[int(s)] for s in subjects], dtype=np.float64)


def fit_weighted_proba(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str],
                       y_train: np.ndarray, sample_weight: np.ndarray, scaler_name: str,
                       C: float = 1.0) -> np.ndarray:
    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
    X_tr = scaler.fit_transform(train_df[feat_cols].to_numpy(dtype=np.float32))
    X_te = scaler.transform(test_df[feat_cols].to_numpy(dtype=np.float32))
    mdl = LogisticRegression(C=C, class_weight="balanced", max_iter=4000,
                              solver="lbfgs", random_state=SEED)
    mdl.fit(X_tr, y_train, sample_weight=sample_weight)
    return mdl.predict_proba(X_te).astype(np.float32)


def exact12_arm_predictions(df_sub: pd.DataFrame, arm_scores: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df_sub), dtype=int)
    for subject in sorted(df_sub["subject"].unique()):
        mask = (df_sub["subject"] == subject).to_numpy()
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(-arm_scores[mask])]
        arm_idx = set(order[:12].tolist())
        pred[mask] = [0 if i in arm_idx else 1 for i in idx]
    return pred


def run_one(df: pd.DataFrame, feat_cols: list[str], feature_set: str, mode: str,
             reactivity: pd.Series, scaler: str, calibration: str, k_hard: int) -> dict:
    df_norm = subject_z_norm(df, feat_cols)
    train_df = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val_df = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    feature_sets = stage2_feature_sets(train_df, feat_cols)
    selected = feature_sets[feature_set]

    train_pain = train_df[train_df["class"].isin(ARM_HAND)].reset_index(drop=True)
    val_pain = val_df[val_df["class"].isin(ARM_HAND)].reset_index(drop=True)

    train_probs = np.zeros((len(train_pain), 2), dtype=np.float32)
    for subj in sorted(train_pain["subject"].unique()):
        tr = train_pain[train_pain["subject"] != subj].reset_index(drop=True)
        te = train_pain[train_pain["subject"] == subj].reset_index(drop=True)
        w = build_weights(tr["subject"].to_numpy(), reactivity, mode, k_hard)
        probs = fit_weighted_proba(tr, te, selected, armhand_binary(tr["class"]),
                                    sample_weight=w, scaler_name=scaler)
        train_probs[train_pain["subject"] == subj] = probs

    w_val = build_weights(train_pain["subject"].to_numpy(), reactivity, mode, k_hard)
    val_probs = fit_weighted_proba(train_pain, val_pain, selected,
                                    armhand_binary(train_pain["class"]),
                                    sample_weight=w_val, scaler_name=scaler)

    cal = fit_binary_calibrator(calibration, train_probs[:, 0],
                                 (train_pain["class"] == "PainArm").astype(int).to_numpy())
    train_arm = cal(train_probs[:, 0])
    val_arm = cal(val_probs[:, 0])

    results = {"mode": mode}
    per_subj_rows = []
    for split_name, sub, scores in (("loso", train_pain, train_arm), ("val", val_pain, val_arm)):
        y_true = (sub["class"] == "PainHand").astype(int).to_numpy()
        y_pred = exact12_arm_predictions(sub, scores)
        m = metrics_binary(y_true, y_pred)
        results[f"{split_name}_macro_f1"] = m["macro_f1"]
        sub_f1s = []
        for subj in sorted(sub["subject"].unique()):
            mask = (sub["subject"] == subj).to_numpy()
            mm = metrics_binary(y_true[mask], y_pred[mask])
            sub_f1s.append(mm["macro_f1"])
            per_subj_rows.append({
                "mode": mode, "split": split_name, "subject": int(subj),
                "reactivity": float(reactivity.get(int(subj), np.nan)),
                "macro_f1": mm["macro_f1"],
            })
        results[f"{split_name}_per_subject_std"] = float(np.std(sub_f1s))
        results[f"{split_name}_per_subject_min"] = float(np.min(sub_f1s))

    # Hard-10 subgroup metric on LOSO
    train_subjects = sorted(train_pain["subject"].unique().tolist())
    r_train = reactivity.reindex(train_subjects).sort_values()
    hard = set(r_train.head(k_hard).index.astype(int).tolist())
    easy = set(r_train.tail(k_hard).index.astype(int).tolist())
    loso_sub = pd.DataFrame([r for r in per_subj_rows if r["split"] == "loso"])
    results["loso_hard10_macro_f1"] = float(loso_sub[loso_sub["subject"].isin(hard)]["macro_f1"].mean())
    results["loso_easy10_macro_f1"] = float(loso_sub[loso_sub["subject"].isin(easy)]["macro_f1"].mean())

    return {"summary": results, "per_subject": pd.DataFrame(per_subj_rows)}


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, feat_cols = load_clean_features(args.feature_parquet)
    train_subjects = sorted(df[df["split"] == "train"]["subject"].unique().tolist())
    print(f"[load] rows={len(df)} train_subj={len(train_subjects)} feats={len(feat_cols)}")

    reactivity = compute_reactivity(df, feat_cols, train_subjects)
    print(f"[reactivity] train median={reactivity.loc[train_subjects].median():.2f} "
          f"range=[{reactivity.loc[train_subjects].min():.2f}, "
          f"{reactivity.loc[train_subjects].max():.2f}]")

    rows = []
    per_subj_frames = []
    for mode in args.modes:
        print(f"\n[run] mode={mode}")
        res = run_one(df, feat_cols, args.feature_set, mode, reactivity,
                       args.scaler, args.calibration, args.k_hard)
        s = res["summary"]
        print(f"    loso={s['loso_macro_f1']:.4f} val={s['val_macro_f1']:.4f} "
              f"loso_std={s['loso_per_subject_std']:.3f} loso_min={s['loso_per_subject_min']:.3f} "
              f"hard10={s['loso_hard10_macro_f1']:.4f} easy10={s['loso_easy10_macro_f1']:.4f}")
        rows.append(s)
        per_subj_frames.append(res["per_subject"])

    summary = pd.DataFrame(rows).sort_values("loso_macro_f1", ascending=False)
    per_subj = pd.concat(per_subj_frames, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_subj.to_csv(out_dir / "per_subject.csv", index=False)

    (out_dir / "report.md").write_text("\n".join([
        "# Stage 2 Weighted Training Sweep",
        "",
        f"- feature_set: `{args.feature_set}`  scaler: `{args.scaler}`  cal: `{args.calibration}`",
        f"- norm: subject_z (fixed)",
        f"- k_hard: {args.k_hard} (bottom by reactivity)",
        "",
        "## Results", "",
        summary.to_markdown(index=False, floatfmt=".4f"),
    ]))
    print(f"\n[done] wrote {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature-parquet", default="results/tables/all_features_merged_1022.parquet")
    p.add_argument("--feature-set", default="resp_all",
                    choices=["resp_all", "resp_top20", "resp_bvp5",
                             "eda_resp_top30", "bvp_resp_top30", "all_top40"])
    p.add_argument("--modes", nargs="+", default=["uniform", "inv_reactivity",
                                                   "squared_inv", "boost_hard", "equal_subject"],
                    choices=["uniform", "inv_reactivity", "squared_inv", "boost_hard", "equal_subject"])
    p.add_argument("--scaler", default="robust", choices=["std", "robust"])
    p.add_argument("--calibration", default="isotonic", choices=["none", "sigmoid", "isotonic"])
    p.add_argument("--k-hard", type=int, default=10)
    p.add_argument("--output-dir", default="results/final/stage2_weighted")
    main(p.parse_args())
