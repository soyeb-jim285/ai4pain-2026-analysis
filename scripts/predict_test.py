from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.final_pipeline import (
    apply_subject_robust, apply_subject_z, build_norm_map,
    decode_joint_weighted, load_clean_features,
)


def main(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    s1 = joblib.load(model_dir / "stage1.joblib")
    s2 = joblib.load(model_dir / "stage2.joblib")

    # Load test features - assume same format as train
    df, feat_cols = load_clean_features(args.test_parquet)
    norm_map = build_norm_map(df, feat_cols)
    s1_df = norm_map[cfg["stage1_norm"]]
    s2_df = norm_map[cfg["stage2_norm"]]

    # Predict stage 1
    X1 = s1["scaler"].transform(s1_df[s1["features"]].to_numpy(dtype=np.float32))
    p1 = s1["model"].predict_proba(X1).astype(np.float32)
    nop_cal = s1["calibrator"](p1[:, 0])
    s1_probs = np.column_stack([nop_cal, 1 - nop_cal]).astype(np.float32)

    # Predict stage 2
    X2 = s2["scaler"].transform(s2_df[s2["features"]].to_numpy(dtype=np.float32))
    p2 = s2["model"].predict_proba(X2).astype(np.float32)
    arm_cal = s2["calibrator"](p2[:, 0])
    s2_probs = np.column_stack([arm_cal, 1 - arm_cal]).astype(np.float32)

    # Decode per-subject exact 12/12/12
    y_pred = np.zeros(len(s1_df), dtype=int)
    for subj in sorted(s1_df["subject"].unique()):
        mask = (s1_df["subject"] == subj).to_numpy()
        y_pred[mask] = decode_joint_weighted(s1_probs[mask], s2_probs[mask],
                                              w0=cfg["w0"], w1=cfg["w1"], w2=cfg["w2"])

    out = s1_df[["subject", "segment_id"]].copy()
    out["pred_y"] = y_pred
    out["pred_class"] = ["NoPain" if i == 0 else ("PainArm" if i == 1 else "PainHand") for i in y_pred]
    out["stage1_nopain_prob"] = s1_probs[:, 0]
    out["stage2_arm_prob"] = s2_probs[:, 0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Saved predictions: {args.output}")
    print(f"Counts: {out['pred_class'].value_counts().to_dict()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--test-parquet", required=True)
    p.add_argument("--output", default="results/final/test_predictions.parquet")
    main(p.parse_args())
