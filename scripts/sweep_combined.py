from __future__ import annotations
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm


def parse_line(lines: list[str], prefix: str) -> dict:
    hits = [l for l in lines if l.strip().startswith(prefix)]
    if not hits:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0}
    parts = hits[0].split()
    return {"accuracy": float(parts[1]), "balanced_accuracy": float(parts[2]), "macro_f1": float(parts[3])}


def run_one(args_list: list[str]) -> dict:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/run_combined.py"] + args_list,
        capture_output=True, text=True,
    )
    lines = result.stdout.strip().splitlines()
    loso = parse_line(lines, "train_loso")
    val = parse_line(lines, "validation")
    return {
        "loso_acc": loso["accuracy"], "loso_f1": loso["macro_f1"],
        "val_acc": val["accuracy"], "val_f1": val["macro_f1"],
    }


def main() -> None:
    resamples = ["1022", "poly1022"]
    s1_feats = ["all", "all_raw"]
    s1_norms = ["subject_z", "subject_robust"]
    s2_feats = ["resp_all", "resp_top20"]
    s2_models = ["xgb", "logreg"]
    s2_cals = ["isotonic", "sigmoid", "none"]
    weights = [(0.8, 1.2, 1.4), (1.0, 1.0, 1.0), (0.6, 1.2, 1.4)]

    configs = [
        (r, sf, sn, s2f, s2m, s2c, w)
        for r in resamples
        for sf in s1_feats
        for sn in s1_norms
        for s2f in s2_feats
        for s2m in s2_models
        for s2c in s2_cals
        for w in weights
    ]
    total = len(configs)
    rows = []
    out_root = Path("results/final/sweep")
    out_root.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(configs, total=total, desc="sweep", ncols=120)
    for r, sf, sn, s2f, s2m, s2c, (w0, w1, w2) in pbar:
        pbar.set_postfix_str(f"r={r} s1={sf}/{sn} s2={s2m}/{s2c} w=({w0},{w1},{w2})")
        flags = [
            "--resample-tag", r,
            "--stage1-norm", sn,
            "--stage1-feature-set", sf,
            "--stage1-model", "xgb",
            "--stage1-scaler", "std",
            "--stage1-calibration", "none",
            "--stage1-anchor-mode", "none",
            "--stage2-norm", "subject_robust",
            "--stage2-feature-set", s2f,
            "--stage2-model", s2m,
            "--stage2-scaler", "robust",
            "--stage2-calibration", s2c,
            "--stage2-anchor-mode", "none",
            "--decoder", "joint_weighted",
            "--w0", str(w0), "--w1", str(w1), "--w2", str(w2),
            "--output-dir", str(out_root / f"{r}_{sf}_{sn}_{s2f}_{s2m}_{s2c}_{w0}_{w2}"),
        ]
        res = run_one(flags)
        row = {
            "resample": r, "s1_feat": sf, "s1_norm": sn,
            "s2_feat": s2f, "s2_model": s2m, "s2_cal": s2c,
            "w0": w0, "w1": w1, "w2": w2, **res,
        }
        rows.append(row)
        tqdm.write(
            f"  {r} s1={sf}/{sn} s2={s2f}/{s2m}/{s2c} w=({w0},{w1},{w2})  "
            f"LOSO f1={res['loso_f1']:.4f}  VAL f1={res['val_f1']:.4f}"
        )
        pd.DataFrame(rows).sort_values("val_f1", ascending=False).to_csv(out_root / "sweep_results.csv", index=False)

    df = pd.DataFrame(rows).sort_values("val_f1", ascending=False)
    print("\nTop 15 by val_f1:")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
