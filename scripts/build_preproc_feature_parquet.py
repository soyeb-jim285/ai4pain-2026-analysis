"""Build a merged feature parquet with a named PreprocConfig applied.

Produces `results/tables/all_features_merged_{resample_tag}_{suffix}.parquet`
(plus per-source tables) compatible with run_combined / best_mission.

Usage:
  uv run python scripts/build_preproc_feature_parquet.py \
      --config eda_detrend_lp1 --suffix edadtlp1 --workers 10

Then best_mission / run_combined with `--resample-tag 1022_edadtlp1` reads it.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import SFREQ, SIGNALS, load_split  # noqa: E402
from src.preprocessing import apply_config  # noqa: E402
from scripts.preproc_configs import get_config  # noqa: E402

warnings.filterwarnings("ignore")

SCRIPT_DIR = ROOT / "scripts"
TAB_DIR = ROOT / "results" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

TARGET_N_SAMPLES = 1022
META_COLS = ["split", "subject", "class", "segment_idx", "segment_id"]


def _load_module(fname: str, modname: str):
    path = SCRIPT_DIR / fname
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tf_row(i, tensor_p, meta_row, tf_mod):
    import warnings as _w
    _w.filterwarnings("ignore")
    np.seterr(all="ignore")
    feats: dict = {}
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        for s_i, sig in enumerate(SIGNALS):
            try:
                feats.update(tf_mod.compute_all(tensor_p[i, s_i], SFREQ, sig))
            except Exception:
                pass
    return {**meta_row, **feats}


def extract_feature_table(
    tensor_p: np.ndarray,
    meta: pd.DataFrame,
    raw_mod, physio_mod, tf_mod,
    split: str,
    n_jobs: int = 1,
) -> pd.DataFrame:
    raw_df = raw_mod.per_segment_stats(tensor_p, meta)
    physio_df = physio_mod.extract_all(tensor_p, meta)
    meta_rows = []
    for i in range(len(meta)):
        row = {}
        for c in META_COLS:
            v = meta.iloc[i][c]
            if c in ("subject", "segment_idx"):
                v = int(v)
            row[c] = v
        meta_rows.append(row)
    if n_jobs == 1:
        tf_rows = [_tf_row(i, tensor_p, meta_rows[i], tf_mod)
                   for i in tqdm(range(len(meta)), desc=f"tf {split}", ncols=80)]
    else:
        from joblib import Parallel, delayed
        tf_rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_tf_row)(i, tensor_p, meta_rows[i], tf_mod)
            for i in tqdm(range(len(meta)), desc=f"tf {split}", ncols=80)
        )
    tf_df = pd.DataFrame(tf_rows)

    raw_feats = [c for c in raw_df.columns if c not in META_COLS]
    phys_feats = [c for c in physio_df.columns if c not in META_COLS]
    tf_feats = [c for c in tf_df.columns if c not in META_COLS]
    collisions = (set(raw_feats) & set(phys_feats)) | (set(raw_feats) & set(tf_feats))
    raw_renamed = raw_df.rename(columns={c: f"raw_{c}" for c in collisions})
    merged = physio_df.merge(
        tf_df.drop(columns=[c for c in META_COLS if c != "segment_id"]),
        on="segment_id", how="inner",
    )
    merged = merged.merge(
        raw_renamed.drop(columns=[c for c in META_COLS if c != "segment_id"]),
        on="segment_id", how="inner",
    )
    return raw_df, physio_df, tf_df, merged


def main(args: argparse.Namespace) -> None:
    cfg = get_config(args.config)
    tag = f"{TARGET_N_SAMPLES}_{args.suffix}"
    print(f"[build] config={cfg.name} suffix={args.suffix} tag={tag}")
    print(f"[build] ops: {cfg.ops}")

    raw_mod = _load_module("02_raw_stats.py", f"prpf_raw_{args.suffix}")
    physio_mod = _load_module("03_physio_features.py", f"prpf_physio_{args.suffix}")
    tf_mod = _load_module("04_tfdomain_features.py", f"prpf_tf_{args.suffix}")

    if args.strip_bvp_filter:
        _orig = physio_mod._butter_filtfilt
        def _pat(x, low, high, sfreq, order=3):
            if low == 0.5 and high == 10.0 and order == 3:
                return x - float(np.mean(x))
            return _orig(x, low, high, sfreq, order)
        physio_mod._butter_filtfilt = _pat
        print("[build] physio BVP internal bandpass stripped")

    raw_frames, physio_frames, tf_frames = [], [], []
    splits = tuple(args.splits.split(",")) if args.splits else ("train", "validation")
    for split in splits:
        t0 = time.time()
        print(f"\n[{split}] loading tensor ...")
        tensor, meta = load_split(split, n_samples=TARGET_N_SAMPLES)
        print(f"[{split}] tensor shape={tensor.shape}")

        print(f"[{split}] applying config {cfg.name} ...")
        tensor_p = apply_config(tensor, cfg, n_jobs=args.workers)

        print(f"[{split}] extracting features ...")
        r, p, t, _ = extract_feature_table(
            tensor_p, meta, raw_mod, physio_mod, tf_mod, split, n_jobs=args.workers,
        )
        raw_frames.append(r)
        physio_frames.append(p)
        tf_frames.append(t)
        print(f"[{split}] done in {time.time()-t0:.1f}s")

    raw_df = pd.concat(raw_frames, ignore_index=True)
    physio_df = pd.concat(physio_frames, ignore_index=True)
    tf_df = pd.concat(tf_frames, ignore_index=True)

    raw_feats = [c for c in raw_df.columns if c not in META_COLS]
    phys_feats = [c for c in physio_df.columns if c not in META_COLS]
    tf_feats = [c for c in tf_df.columns if c not in META_COLS]
    collisions = (set(raw_feats) & set(phys_feats)) | (set(raw_feats) & set(tf_feats))
    raw_renamed = raw_df.rename(columns={c: f"raw_{c}" for c in collisions})
    merged_df = physio_df.merge(
        tf_df.drop(columns=[c for c in META_COLS if c != "segment_id"]),
        on="segment_id", how="inner",
    )
    merged_df = merged_df.merge(
        raw_renamed.drop(columns=[c for c in META_COLS if c != "segment_id"]),
        on="segment_id", how="inner",
    )

    out_merged = TAB_DIR / f"all_features_merged_{tag}.parquet"
    out_raw = TAB_DIR / f"raw_stats_per_segment_{tag}.parquet"
    out_physio = TAB_DIR / f"physio_features_{tag}.parquet"
    out_tf = TAB_DIR / f"tf_features_{tag}.parquet"

    raw_df.to_parquet(out_raw, index=False)
    physio_df.to_parquet(out_physio, index=False)
    tf_df.to_parquet(out_tf, index=False)
    merged_df.to_parquet(out_merged, index=False)

    print(f"\n[done]")
    print(f"  merged  shape={merged_df.shape}  -> {out_merged}")
    print(f"  raw     shape={raw_df.shape}     -> {out_raw}")
    print(f"  physio  shape={physio_df.shape}  -> {out_physio}")
    print(f"  tf      shape={tf_df.shape}      -> {out_tf}")
    print(f"\n[next] best_mission with: --resample-tag {tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="PreprocConfig name (e.g. eda_detrend_lp1)")
    p.add_argument("--suffix", required=True, help="parquet tag suffix (e.g. edadtlp1)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--strip-bvp-filter", action="store_true")
    p.add_argument("--splits", default=None, help="comma-sep splits (default: train,validation)")
    main(p.parse_args())
