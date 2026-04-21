"""
Best-mission combined pipeline.

Runs the winning two-stage NoPain/PainArm/PainHand pipeline with hardcoded
defaults found via stage-1 and stage-2 sweeps (2026-04-21 session).

Expected LOSO 3-class macro-F1 ~= 0.589 +- 0.128 (n=41 train subjects).
Expected validation 3-class macro-F1 ~= 0.574 +- 0.084 (n=12 val subjects).

Config:
  features       : truncate1022
  stage1 norm    : subject_z
  stage1 feats   : bvp_eda_resp_small (70)
  stage1 model   : logreg (C=1.0, class_weight=balanced)
  stage1 scaler  : std
  stage1 cal     : isotonic
  stage1 anchor  : none
  stage2 norm    : subject_z
  stage2 feats   : bvp_resp_top30 (30)
  stage2 model   : logreg (C=1.0, class_weight=balanced)
  stage2 scaler  : robust
  stage2 cal     : isotonic
  stage2 anchor  : none
  decoder        : joint_weighted
  weights        : w0=1.0  w1=1.0  w2=1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_combined import main as run_combined_main


BEST_CONFIG = {
    "feature_parquet": None,
    "resample_tag": "1022",

    "stage1_norm": "subject_z",
    "stage1_feature_set": "bvp_eda_resp_small",
    "stage1_model": "logreg",
    "stage1_scaler": "std",
    "stage1_calibration": "isotonic",
    "stage1_anchor_mode": "none",
    "stage1_anchor_lambda": 0.5,
    "stage1_xgb_n_estimators": 200,
    "stage1_xgb_max_depth": 4,
    "stage1_xgb_learning_rate": 0.08,
    "stage1_xgb_subsample": 1.0,
    "stage1_xgb_colsample_bytree": 1.0,
    "stage1_rf_n_estimators": 400,
    "stage1_rf_max_depth": None,
    "stage1_logreg_c": 1.0,
    "stage1_svm_c": 3.0,
    "stage1_svm_gamma": "scale",

    "stage2_norm": "subject_z",
    "stage2_feature_set": "bvp_resp_top30",
    "stage2_model": "logreg",
    "stage2_scaler": "robust",
    "stage2_calibration": "isotonic",
    "stage2_anchor_mode": "none",
    "stage2_xgb_n_estimators": 200,
    "stage2_xgb_max_depth": 4,
    "stage2_xgb_learning_rate": 0.08,
    "stage2_xgb_subsample": 1.0,
    "stage2_xgb_colsample_bytree": 1.0,
    "stage2_rf_n_estimators": 400,
    "stage2_rf_max_depth": None,
    "stage2_logreg_c": 1.0,
    "stage2_svm_c": 3.0,
    "stage2_svm_gamma": "scale",

    "decoder": "joint_weighted",
    "w0": 1.0,
    "w1": 1.0,
    "w2": 1.0,

    "output_dir": "results/final/best_mission",
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Best-mission combined pipeline (hardcoded winning defaults).")
    p.add_argument("--output-dir", default=BEST_CONFIG["output_dir"],
                    help="override output directory")
    p.add_argument("--resample-tag", default=BEST_CONFIG["resample_tag"],
                    help="feature parquet tag (1022 / linear1022 / poly1022)")
    args = p.parse_args()

    ns = argparse.Namespace(**BEST_CONFIG)
    ns.output_dir = args.output_dir
    ns.resample_tag = args.resample_tag
    run_combined_main(ns)


if __name__ == "__main__":
    main()
