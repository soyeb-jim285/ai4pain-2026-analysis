from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.final_pipeline import class_codes_3, decode_joint_weighted, metrics_multiclass, plot_confusion


def main(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.predictions_parquet)
    sub = df[df["config_id"] == args.config_id].copy().sort_values(["subject", "segment_id"]).reset_index(drop=True)
    p_pain = sub["stage1_pain_prob"].to_numpy()
    p_arm = sub["stage2_arm_prob"].to_numpy()
    s1 = np.column_stack([1 - p_pain, p_pain])
    s2 = np.column_stack([p_arm, 1 - p_arm])
    y_pred = np.zeros(len(sub), dtype=int)
    for subj in sorted(sub["subject"].unique()):
        mask = (sub["subject"] == subj).to_numpy()
        y_pred[mask] = decode_joint_weighted(s1[mask], s2[mask], w0=args.w0, w1=args.w1, w2=args.w2)
    y_true = class_codes_3(sub["class"])
    met = metrics_multiclass(y_true, y_pred)
    print(f"config: {args.config_id}")
    print(f"weights: w0={args.w0} w1={args.w1} w2={args.w2}")
    for k, v in met.items():
        print(f"  {k}: {v:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub["pred_y"] = y_pred
    sub["true_y"] = y_true
    sub.to_parquet(out_dir / "decoded.parquet", index=False)
    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
    plot_confusion(cm, ["NoPain", "Arm", "Hand"], "Cached Suite31 Decode", out_dir / "confusion.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-parquet", default="results/tables/suite31_validation_predictions.parquet")
    p.add_argument("--config-id", default="truncate1022|xgb_subject_z|resp_all_robust|joint_weighted")
    p.add_argument("--w0", type=float, default=0.8)
    p.add_argument("--w1", type=float, default=1.2)
    p.add_argument("--w2", type=float, default=1.4)
    p.add_argument("--output-dir", default="results/final/cached_suite31_decode")
    main(p.parse_args())
