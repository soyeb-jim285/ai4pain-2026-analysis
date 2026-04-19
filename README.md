# AI4Pain 2026 — analysis workspace

Subject-independent pain detection and localisation from multimodal physiological
signals. Accompanying analysis code, feature extraction pipelines, statistical
tests, visualisations and baseline models for the AI4Pain 2026 challenge dataset.

## Data

The dataset itself is **not** included in this repository (see `.gitignore`).
Place the challenge data under `Dataset/` so the structure is:

```
Dataset/
├── train/
│   ├── Bvp/{subject}.csv
│   ├── Eda/{subject}.csv
│   ├── Resp/{subject}.csv
│   └── SpO2/{subject}.csv
└── validation/
    └── (same layout)
```

- 4 signals per subject, 100 Hz sampling.
- Each CSV has 36 columns: 12 Baseline + 12 ARM + 12 HAND segments (~10 s each).
- 41 train subjects, 12 validation subjects. Classes are balanced 12/12/12 per subject.

See `AI4Pain 2026 Challenge - Experimental Procedure, Dataset, and Participants.pdf`
(if shipped alongside the repo) for the full description.

## Project layout

```
.                                  # repo root (was analysis/)
├── pyproject.toml                 # uv project, Python ≥3.10
├── uv.lock
├── src/
│   └── data_loader.py             # reusable loader — load_split(split) -> (tensor, meta)
├── scripts/
│   ├── 00_prime_cache.py          # build numpy + parquet caches
│   ├── 01_inventory.py            # integrity + range checks
│   ├── 02_raw_stats.py            # per-segment descriptive stats
│   ├── 03_physio_features.py      # HR / HRV / SCR / breathing / SpO2 features
│   ├── 04_tfdomain_features.py    # time & frequency (158 features)
│   ├── 05_visualizations.py       # 13 exploratory figures
│   ├── 06_class_tests.py          # ANOVA / Wilcoxon / Cliff's δ + FDR
│   ├── 07_dimreduction.py         # PCA / t-SNE / UMAP × 3 preprocessings
│   ├── 08_baselines.py            # LOSO baselines (3-class / binary / arm-vs-hand)
│   ├── 09_temporal_armhand.py     # 2-s sub-window trajectories
│   ├── 10_morphology_coupling.py  # PPG morphology + cross-signal coupling
│   ├── 11_reactivity_interactions.py
│   ├── 12_armhand_stronger_ml.py  # LR / RF / XGB / sub-window / 1-D CNN
│   ├── 13_tierA1_onset_alignment.py
│   ├── 14_tierA2_bvp_kinetics.py
│   ├── 15_tierA3_stage1_calibration.py
│   ├── 16_tierA4_subject_clusters.py
│   ├── 17_tierA5_channel_restricted.py
│   └── 18_armhand_search.py       # resumable hyperparameter sweep
├── results/
│   ├── tables/                    # CSV + parquet deliverables
│   └── reports/                   # per-stage markdown summaries + FINDINGS.md
├── plots/                         # all PNG figures
├── cache/                         # primed *.npz / *.parquet (gitignored)
└── Dataset/                       # raw challenge data (gitignored)
```

## Quick start

```bash
uv sync
uv run python scripts/00_prime_cache.py
uv run python scripts/01_inventory.py
# ... and so on for each stage
```

Each stage writes to `results/tables/`, `results/reports/` and `plots/`. Stages
depend only on the cached tensors and earlier feature tables — not on each
other's plots — so they can be re-run individually.

## Key findings at a glance

Full story: [`analysis/results/reports/FINDINGS.md`](analysis/results/reports/FINDINGS.md).

| task | best LOSO macro-F1 | validation macro-F1 | chance |
|---|---|---|---|
| No-Pain vs Pain | 0.819 ± 0.11 (XGB + subject-z) | 0.801 | 0.50 |
| 3-class | 0.575 ± 0.11 (LogReg + subject-z) | 0.515 (best val 0.552) | 0.33 |
| Arm vs Hand | 0.559 ± 0.10 (LogReg + subject-z) | ~0.51 (near chance) | 0.50 |

Three things worth knowing:

1. **Subject identity dominates.** Without per-subject z-scoring, a 10-NN search
   returns the same subject ~76% of the time. Always normalise per-subject.
2. **Pain detection works (~0.80 macro-F1); pain localisation doesn't.** Out of
   646 engineered features, 0 survive FDR<0.05 for PainArm vs PainHand at the
   subject-mean level. Only 4 mid-segment RESP shape features reach FDR<0.05
   under a segment-level mixed-effects model.
3. **Data quirks:** 349 SpO₂ segments are flat (dead sensor), raw segment
   lengths are {990, 1022, 1118} (not 1000), and subject 46 should be inspected
   (25 % LOSO accuracy suggests a per-subject artefact).

## Reproducibility

- `uv sync` pins exact package versions via `uv.lock`.
- All scripts accept the default paths; results should reproduce byte-for-byte
  given the same dataset.
- Random seeds are fixed to 42 where relevant.
