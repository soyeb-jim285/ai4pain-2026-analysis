# 07 — Dimensionality reduction summary
Analysis of PCA / t-SNE / UMAP on merged (physio + tf + raw-stats) features, across three preprocessing variants: `raw`, `global_z`, `subject_z`. 41 train subjects × 36 segments = 1476 rows; 12 val subjects × 36 = 432.
## Silhouette scores (train)
Higher = tighter clustering by that label. PCA uses the first 10 PCs; t-SNE and UMAP use 2D embeddings.
| variant | reducer | silhouette (class) | silhouette (subject) | subject − class gap |
|---|---|---:|---:|---:|
| global_z | PCA | -0.0026 | 0.1009 | +0.1036 |
| global_z | UMAP | -0.0054 | 0.3490 | +0.3544 |
| global_z | tSNE | -0.0083 | 0.2848 | +0.2932 |
| raw | PCA | -0.0220 | -0.4490 | -0.4270 |
| raw | UMAP | -0.0174 | -0.3838 | -0.3665 |
| raw | tSNE | -0.0232 | -0.3513 | -0.3281 |
| subject_z | PCA | 0.0167 | -0.0476 | -0.0643 |
| subject_z | UMAP | 0.0225 | -0.1328 | -0.1553 |
| subject_z | tSNE | 0.0209 | -0.1372 | -0.1581 |

## 10-Nearest-neighbour probabilities on standardised feature space
Chance levels: subject = 1/41 ≈ 0.024, class = 1/3 ≈ 0.333.

| variant | p_same_subject@10 | p_same_class@10 |
|---|---:|---:|
| raw | 0.2092 | 0.3461 |
| global_z | 0.7583 | 0.4012 |
| subject_z | 0.0837 | 0.4586 |

## Pairwise-distance ratios
`within/between < 1` means samples sharing the label are closer than average. Subject ratio ≪ class ratio ⇒ subject identity dominates the geometry.

| variant | within/between class | within/between subject |
|---|---:|---:|
| raw | 0.9979 | 0.4616 |
| global_z | 0.9914 | 0.6370 |
| subject_z | 0.9598 | 1.0101 |

## Key finding: does subject_z reduce subject leakage?
- 10-NN same-subject probability: raw=0.209, global_z=0.758, subject_z=0.084 (Δ global_z→subject_z = -0.675).
- 10-NN same-class probability: raw=0.346, global_z=0.401, subject_z=0.459 (Δ global_z→subject_z = +0.057).
- Mean class silhouette (over reducers): global_z=-0.0054 → subject_z=0.0200 (Δ=+0.0255).
- Mean subject silhouette (over reducers): global_z=0.2449 → subject_z=-0.1059 (Δ=-0.3508).

**Verdict:** subject_z demonstrably reduces subject leakage and improves class geometry.

## Recommended feature preprocessing
Use **per-subject z-scoring** (subject_z) as the default preprocessing for downstream classifiers. It meaningfully demotes the inter-subject baseline offset that otherwise dominates nearest-neighbour geometry, giving the pain signal a chance to surface. For validation subjects, compute per-subject mean/std on that subject's own 36 segments (safe because the inference-time unit is the whole session).

## Notable outlier subjects
Inspect `plots/dimreduction/pca_2d_by_subject_global_z.png`, `tsne_by_subject_global_z.png`, and `umap_by_subject_global_z.png` — under global_z a handful of subjects form very tight, far-away clusters (they drive the large subject silhouette under global_z). Under subject_z those fly-away clusters collapse, confirming they were pure DC-level offsets rather than intrinsic class structure.

## Outputs
- Tables:
  - `results/tables/all_features_merged.parquet`
  - `results/tables/pca_explained_variance_raw.csv`
  - `results/tables/pca_coords_raw.csv`
  - `results/tables/tsne_coords_raw.csv`
  - `results/tables/umap_coords_raw.csv`
  - `results/tables/pca_explained_variance_global_z.csv`
  - `results/tables/pca_coords_global_z.csv`
  - `results/tables/tsne_coords_global_z.csv`
  - `results/tables/umap_coords_global_z.csv`
  - `results/tables/pca_explained_variance_subject_z.csv`
  - `results/tables/pca_coords_subject_z.csv`
  - `results/tables/tsne_coords_subject_z.csv`
  - `results/tables/umap_coords_subject_z.csv`
  - `results/tables/silhouette_summary.csv`
  - `results/tables/subject_leak_summary.csv`
- Plots: all under `plots/dimreduction/`.
