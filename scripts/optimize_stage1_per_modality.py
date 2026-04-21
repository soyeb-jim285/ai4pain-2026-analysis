"""Stage 1 per-modality optimization: ranked top-K sweep + greedy forward selection."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.final_pipeline import (
    ARM_HAND, ModelSpec, apply_subject_robust, channel_of, fit_binary_calibrator,
    fit_binary_proba, load_clean_features, rank_binary_features, unique_by_canonical,
)

OUT = Path("results/final/stage1_optim")
OUT.mkdir(parents=True, exist_ok=True)

MOD_POOLS = {
    "BVP": ["bvp"],
    "EDA": ["eda"],
    "RESP": ["resp"],
    "SpO2": ["spo2"],
    "EDA+BVP": ["eda", "bvp"],
    "EDA+BVP+RESP": ["eda", "bvp", "resp"],
    "All": ["bvp", "eda", "resp", "spo2"],
}

K_VALUES = [5, 10, 15, 20, 30, 50, 75, 100]


def select_features(feat_cols, tags):
    return unique_by_canonical([c for c in feat_cols if channel_of(c) in tags])


def loso_val_auc(df_norm, feats, spec):
    train = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    val = df_norm[df_norm["split"] == "validation"].reset_index(drop=True)
    # LOSO probs
    tp = np.zeros((len(train), 2), dtype=np.float32)
    for s in sorted(train["subject"].unique()):
        tr = train[train["subject"] != s].reset_index(drop=True)
        te = train[train["subject"] == s].reset_index(drop=True)
        tp[train["subject"] == s] = fit_binary_proba(
            tr, te, feats, (tr["class"] != "NoPain").astype(int).to_numpy(), "std", spec
        )
    vp = fit_binary_proba(train, val, feats, (train["class"] != "NoPain").astype(int).to_numpy(), "std", spec)
    # AUC with Pain as positive
    y_tr = (train["class"] != "NoPain").astype(int).to_numpy()
    y_va = (val["class"] != "NoPain").astype(int).to_numpy()
    loso_auc = roc_auc_score(y_tr, 1 - tp[:, 0])
    val_auc = roc_auc_score(y_va, 1 - vp[:, 0])
    return loso_auc, val_auc


def topk_sweep(df_norm, train_df, feat_cols, tags, ks, spec, mod_name=""):
    pool = select_features(feat_cols, tags)
    ranked = rank_binary_features(train_df, pool, positive="Pain")
    rows = []
    ks_valid = [k for k in ks if k <= len(pool)]
    pbar = tqdm(ks_valid, desc=f"  topK {mod_name}", leave=False, ncols=90)
    for k in pbar:
        feats = ranked[:k]
        loso_auc, val_auc = loso_val_auc(df_norm, feats, spec)
        rows.append({"k": k, "loso_auc": loso_auc, "val_auc": val_auc})
        pbar.set_postfix_str(f"K={k} LOSO={loso_auc:.3f} VAL={val_auc:.3f}")
    return pd.DataFrame(rows), ranked


def greedy_forward(df_norm, train_df, feat_cols, tags, max_k, spec, mod_name="", verbose=False):
    """Greedy forward: start empty, add feature that most improves LOSO AUC."""
    pool = select_features(feat_cols, tags)
    ranked = rank_binary_features(train_df, pool, positive="Pain")
    candidates = list(ranked[:min(40, len(ranked))])
    selected = []
    history = []
    best_auc = 0.0
    outer_pbar = tqdm(range(max_k), desc=f"  fwd {mod_name}", leave=False, ncols=90)
    for step in outer_pbar:
        if not candidates:
            break
        scores = []
        inner_pbar = tqdm(candidates, desc=f"    step {step+1}", leave=False, ncols=80)
        for f in inner_pbar:
            trial = selected + [f]
            try:
                loso_auc, val_auc = loso_val_auc(df_norm, trial, spec)
                scores.append((loso_auc, val_auc, f))
            except Exception:
                continue
        if not scores:
            break
        scores.sort(reverse=True)
        best_l, best_v, best_f = scores[0]
        if best_l < best_auc - 0.005:
            outer_pbar.write(f"    stop: no improvement (best {best_l:.4f} < {best_auc:.4f})")
            break
        selected.append(best_f)
        candidates.remove(best_f)
        best_auc = best_l
        history.append({"k": len(selected), "feat_added": best_f, "loso_auc": best_l, "val_auc": best_v})
        outer_pbar.set_postfix_str(f"k={len(selected)} LOSO={best_l:.3f} VAL={best_v:.3f} +{best_f[:25]}")
        if verbose:
            tqdm.write(f"    +{best_f}: loso={best_l:.4f} val={best_v:.4f}")
    return pd.DataFrame(history), selected


def main():
    df_all, feat_cols = load_clean_features("results/tables/all_features_merged_1022.parquet")
    df_norm = apply_subject_robust(df_all, feat_cols)
    train_df = df_norm[df_norm["split"] == "train"].reset_index(drop=True)
    spec = ModelSpec("xgb", {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0})

    summary = {}
    mod_items = list(MOD_POOLS.items())
    mod_pbar = tqdm(mod_items, desc="modality", ncols=100)
    for mod, tags in mod_pbar:
        pool = select_features(feat_cols, tags)
        if len(pool) < 5:
            continue
        mod_pbar.set_postfix_str(f"{mod} ({len(pool)} feats)")
        tqdm.write(f"\n===== {mod} ({len(pool)} features) =====")

        # Top-K sweep
        topk_df, ranked = topk_sweep(df_norm, train_df, feat_cols, tags, K_VALUES, spec, mod_name=mod)
        topk_df.to_csv(OUT / f"{mod}_topk.csv", index=False)
        tqdm.write("Top-K sweep:")
        tqdm.write(topk_df.to_string(index=False))
        best_topk = topk_df.iloc[topk_df["val_auc"].idxmax()]

        # Greedy forward
        tqdm.write("Greedy forward selection:")
        fwd_df, fwd_feats = greedy_forward(df_norm, train_df, feat_cols, tags, max_k=25, spec=spec, mod_name=mod, verbose=True)
        fwd_df.to_csv(OUT / f"{mod}_forward.csv", index=False)
        best_fwd = fwd_df.iloc[fwd_df["val_auc"].idxmax()] if len(fwd_df) else None

        summary[mod] = {
            "n_pool": len(pool),
            "best_topk_k": int(best_topk["k"]),
            "best_topk_loso_auc": float(best_topk["loso_auc"]),
            "best_topk_val_auc": float(best_topk["val_auc"]),
            "best_fwd_k": int(best_fwd["k"]) if best_fwd is not None else 0,
            "best_fwd_loso_auc": float(best_fwd["loso_auc"]) if best_fwd is not None else 0,
            "best_fwd_val_auc": float(best_fwd["val_auc"]) if best_fwd is not None else 0,
            "top_ranked_features": ranked[:20],
            "forward_selected": fwd_feats[:20],
        }

    # Save summary
    sum_rows = []
    for mod, s in summary.items():
        sum_rows.append({
            "modality": mod,
            "n_pool": s["n_pool"],
            "best_topk_k": s["best_topk_k"],
            "best_topk_loso_auc": s["best_topk_loso_auc"],
            "best_topk_val_auc": s["best_topk_val_auc"],
            "best_fwd_k": s["best_fwd_k"],
            "best_fwd_loso_auc": s["best_fwd_loso_auc"],
            "best_fwd_val_auc": s["best_fwd_val_auc"],
        })
    pd.DataFrame(sum_rows).to_csv(OUT / "summary.csv", index=False)

    # Markdown
    lines = ["# Stage 1 Per-Modality Optimization", ""]
    lines.append("| Modality | Pool | Best TopK (K) | TopK LOSO AUC | TopK VAL AUC | Best Fwd (K) | Fwd LOSO AUC | Fwd VAL AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mod, s in summary.items():
        lines.append(f"| {mod} | {s['n_pool']} | {s['best_topk_k']} | {s['best_topk_loso_auc']:.4f} | {s['best_topk_val_auc']:.4f} | {s['best_fwd_k']} | {s['best_fwd_loso_auc']:.4f} | {s['best_fwd_val_auc']:.4f} |")
    lines.append("")
    for mod, s in summary.items():
        lines.append(f"## {mod} — Top Ranked Features (cliff delta)")
        lines.append("")
        lines.append("```")
        for i, f in enumerate(s["top_ranked_features"], 1):
            lines.append(f"{i:2d}. {f}")
        lines.append("```")
        lines.append("")
        lines.append(f"## {mod} — Greedy Forward Selected")
        lines.append("")
        lines.append("```")
        for i, f in enumerate(s["forward_selected"], 1):
            lines.append(f"{i:2d}. {f}")
        lines.append("```")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n\n=== SUMMARY ===")
    print(pd.DataFrame(sum_rows).to_string(index=False))


if __name__ == "__main__":
    main()
