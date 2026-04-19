# AI4Pain 2026 — Consolidated Findings

*Subject-independent pain detection & localisation from multimodal physiological signals.*

- 65 participants, Train **41** (1476 segments) · Validation **12** (432 segments) · Test (labels withheld).
- Each subject contributes 12 × NoPain + 12 × PainArm + 12 × PainHand ≈ 10 s windows @ 100 Hz.
- 4 channels: **BVP** (µS of vascular pulse), **EDA** (µS), **RESP** (AU), **SpO₂** (%).

Everything below is backed by numbers in `results/tables/*` and figures in `plots/*`. All scripts reusable via `uv run python scripts/XX.py`.

---

## 1. Dataset integrity — what a modeller needs to know before touching the data

| Issue | Evidence | Implication |
|---|---|---|
| **SpO₂ has 349 constant segments** (254/1476 train, 95/432 val) with std≈0, many pinned at 95–100 %, some at 0–2 % | `inventory_constant_segments.csv` | Treat SpO₂ as partially broken. Either mask flatline segments or down-weight the channel. |
| All **duplicate-signal groups are SpO₂** | `inventory_duplicate_segments.csv` | Same sensor issue — no duplicates in BVP/EDA/RESP. |
| Raw segment lengths are **discrete: {990, 1022, 1118}** (train), uniformly 1022 (val) — not 1000 | `inventory_segment_lengths.csv` | Default truncation to 1000 samples drops 22–118 samples from 98.1 % of segments. Consider `n_samples=1022` in `load_split()`. |
| No interior NaN / inf anywhere | `inventory_nan_inf.csv` | Padding NaNs are the only NaNs; strip with `x[~np.isnan(x)]`. |
| No subject overlap between splits; class counts are exactly 12/12/12/subject | `inventory_subject_splits.csv`, `inventory_subjects.csv` | Splits are clean; LOSO on train is valid. |
| BVP values are very narrow (~0.48–0.52 mV, p1–p99 < 0.05 wide) | `inventory_signal_ranges.csv` | Pain-related deltas are small in absolute units — always z-score. |
| EDA has a hard floor at 0 µS | same | Good — consistent with µS semantics. |

---

## 2. The central problem: subject identity **dominates** class identity

This is the single most important finding and shapes every modelling decision below.

**Evidence (from `plots/dimreduction/*` and `subject_leak_summary.csv`):**

| Preprocessing | 10-NN `p(same subject)` | 10-NN `p(same class)` | Mean tSNE class silhouette | Mean tSNE subject silhouette |
|---|---|---|---|---|
| raw | 0.21 | 0.35 | −0.023 | −0.351 |
| **global-z** (standard) | **0.76** | 0.40 | −0.008 | +0.285 |
| **subject-z** (per-subject demean/rescale) | **0.08** | **0.46** | +0.021 | −0.137 |

- Under conventional `StandardScaler`, a 10-NN search returns the *same subject* 76 % of the time — i.e. models trained on global-z features are mostly learning subject identity.
- After per-subject z-scoring (each subject's features demeaned and rescaled by that subject's own std across their 36 segments), subject leakage collapses to 8 %, within-subject/between-subject distance ratio goes from 0.36 → 1.01 (subject clumps dissolve), and class silhouette flips positive for the first time.
- See `plots/viz/group_mean_waveform_per_class.png` vs `plots/viz/group_mean_waveform_per_class_subject_normalized.png` — the raw pooled average is flat, the baseline-normalised version reveals a clean EDA rise and SpO₂ drift during pain segments.

**Modelling rule**: always subject-normalise (or pair within-subject) before any downstream step.

---

## 3. What actually changes during pain — a physiology-first view

All class effects evaluated on **subject-mean features** (41 subjects × 3 classes) with Benjamini–Hochberg FDR at q = 0.05. Raw pooled-segment tests are inflated and should not be trusted.

### 3.1 Pain vs No-Pain (detection)

- **17/243 features significant** at FDR<0.05 (one-way ANOVA).
- **111/243 significant** under paired within-subject Wilcoxon (more powerful because each subject is their own control).
- Typical effect sizes for top features: |Cliff's δ| ≈ **0.69** — large.
- Top discriminators (all "Pain < NoPain"):
  1. `Bvp_rms` / `Bvp_mean` / `Bvp_energy` — pulse-wave amplitude drops during pain (vasoconstriction, sympathetic vasomotor response).
  2. `Eda_bp_higher` — phasic EDA band power rises during pain.
  3. `SpO2_hjorth_complexity` / `SpO2_hjorth_mobility` — SpO₂ trace becomes more jittery/less stationary (partly artefactual on broken SpO₂ channels, interpret with caution).
  4. `Eda_kurtosis` — tail-heavier EDA distribution, consistent with skin-conductance responses (SCRs).
  5. `EDA_tonic_slope` — the only **physiologically-extracted** feature that clears the subject-mean ANOVA threshold alone: NoPain = −0.011 µS/s → PainArm = +0.020 → PainHand = +0.025. Tonic EDA *rises* during pain.

- **Validation consistency**: all 11 features that pass 3-class ANOVA FDR<0.05 preserve direction in validation (11/11).

### 3.2 PainArm vs PainHand (localisation) — the *hard* question

- **0/243 features survive paired-Wilcoxon FDR**<0.05. *This is a real, honest negative result.*
- Smallest raw p-values (~0.003, not FDR-significant) come from RESP morphology: `RESP_amp_std`, `Resp_mad`.
- Classifier macro-F1 on arm-vs-hand (pain-only segments): 0.56 LOSO → 0.50 validation, i.e. **at chance on held-out subjects**.
- Anatomical stimulus location is not recoverable from 10 s of these four channels with aggregate features. Expected from first principles: peripheral autonomic signals respond to pain magnitude, not pain locus; location-specific motor or cortical signals (EMG, EEG, fMRI) would be needed.

### 3.3 Where the signal lives — channel ranking

Blended feature importance across LogReg + RF + XGB (LOSO, both preprocessings) — top classes of features:
1. **EDA spectral features** — bandwidth, dfa_alpha, p90, higher-band power, Hjorth complexity, tonic level.
2. **BVP amplitude features** — p90, diff-std, total power, peak power, mean, energy, RMS.
3. **RESP rate + morphology** — secondary contribution.
4. **SpO₂** — frequently top-rated but partly driven by the flatline-segment artefact; interpret cautiously.

See `plots/baselines/feature_importance_top20.png`.

---

## 4. Baseline classification performance

LOSO on 41 train subjects, then final model trained on all train and evaluated on 12 held-out validation subjects. Chance reference: 3-class = 0.333, binary = 0.500, arm-vs-hand = 0.500.

### 4.1 Headline numbers (macro-F1)

| Task | Best config | LOSO macro-F1 | Validation macro-F1 |
|---|---|---|---|
| **NoPain vs Pain** | XGB + subject-z | **0.819 ± 0.111** | 0.801 |
| **3-class** (NoPain / PainArm / PainHand) | LogReg + subject-z | **0.575 ± 0.110** | 0.515 (best val: RF-subjectz 0.552) |
| **PainArm vs PainHand** | LogReg + subject-z | **0.559 ± 0.104** | 0.510 (best val: XGB-global 0.535 ≈ chance) |

### 4.2 Subject-z provides a large, consistent lift

Mean macro-F1 across models:

| label scheme | global-z | subject-z | Δ |
|---|---|---|---|
| 3-class | 0.43 | **0.56** | +0.12 |
| binary | 0.68 | **0.80** | +0.12 |
| arm-vs-hand | 0.46 | **0.53** | +0.07 |

This is a **~+12 macro-F1-point jump** from one preprocessing change — larger than any model-architecture change I measured. Any downstream model should subject-normalise.

### 4.3 Generalisation gap (LOSO → validation)

Small and well-behaved for most configurations (|gap| < 0.07). The only interesting cases:
- Logreg-subjectz-3class: LOSO 0.575 → val 0.515 (+0.06 gap). Some held-out subjects in validation are unlike any training subject.
- Binary XGB-subjectz: LOSO 0.819 → val 0.801, gap +0.02 — excellent generalisation for pain detection.
- Arm-vs-hand: gaps small but the whole performance band is near chance, so gap interpretation is moot.

### 4.4 Worst-generalising subjects (bottom-5 LOSO accuracy, best 3-class config)

| subject_held_out | accuracy | macro-F1 |
|---|---|---|
| 46 | 0.250 | 0.250 |
| 36 | 0.389 | 0.391 |
| 56 | 0.389 | 0.392 |
| 22 | 0.444 | 0.433 |
| 55 | 0.444 | 0.437 |

Subject 46 in particular should be inspected — per-signal plots and the inventory report may reveal a sensor artefact.

---

## 5. Secondary / surprising observations

- **Subject effect is visible as an *experiment-length drift*.** `plots/viz/segment_position_effect.png` shows mean HR drifting mildly across segment-index 1→12 within each class (habituation / time-on-task). Subject-z absorbs this; global-z does not.
- **Physio pipelines** (HR, SCR, resp rate) were 100 % successful on all 1908 segments — the neurokit2 path never needed the manual fallback. HR median 78.7 bpm, SpO₂ median 93.3 %, resp 17.4 bpm — all physiologically plausible, though SpO₂ P5 = 42 % reflects the dropout-segment issue.
- **Raw-signal ANOVA top-10 is almost entirely BVP-based** (see `raw_stats_anova_quick.csv`). EDA shape stats (kurtosis/skew) beat EDA amplitude at the subject-mean level — because subject-level tonic-level DC dominates EDA amplitude but not shape.
- **Pairwise Wilcoxon has 6× more discoveries than one-way ANOVA** (111 vs 17 features significant for Pain vs NoPain) — using within-subject pairing buys real power. Any statistical test on this dataset should pair within subject where possible.

---

## 6. Recommended modelling strategy for the challenge

1. **Always subject-normalise.** Per-subject z-score of all features across that subject's 36 segments — or, equivalently, subtract each subject's NoPain baseline from their pain segments. Without this you are training a subject identifier.
2. **For the No-Pain vs Pain sub-problem** the current XGB + subject-z baseline already reaches **0.80 macro-F1 on held-out subjects**. Gains from here will come from:
   - Richer 10-s features (wavelet-spectrum, BVP peak-envelope dynamics, HRV non-linear metrics if you can make SampEn robust on 10 s).
   - Ensemble-of-windows within a segment (sub-windowing 2-s, classifying, majority vote).
   - End-to-end 1-D CNN on the 4-channel × 1000-sample tensor with subject-wise normalisation inside the network.
3. **For the Arm-vs-Hand localisation** don't expect more than chance-adjacent performance with these four physiological channels. If the challenge rewards this, look for:
   - Phase-locked respiratory/pulse features (small morphology differences may encode brachial-plexus-specific reflexes).
   - Ipsilateral-vs-contralateral sensor asymmetries — but this dataset only has right-arm stimulation, so there's no contralateral control.
   - Higher-order coupling (BVP–RESP phase, EDA–BVP Granger causality) — speculative.
4. **Hold out flatline SpO₂ segments** (std < 1e-8) from both training and the SpO₂-based features you build. They contaminate any feature that integrates over the window.
5. **Flag outlier subjects** (≥46 for inspection) before committing to heavy training — one bad-sensor subject can pull macro-F1 down 2–3 points by themselves in LOSO.

---

## 7. File map

Everything is under `analysis/`:

```
src/data_loader.py              # reusable loader, `load_split(split) -> (tensor, meta)`
scripts/
  00_prime_cache.py             # loads both splits into npz + parquet cache
  01_inventory.py               # integrity + range checks
  02_raw_stats.py               # per-segment descriptive stats
  03_physio_features.py         # HR / HRV / SCR / breathing-rate / SpO2 features
  04_tfdomain_features.py       # 158 time & frequency features (entropy, fractal, PSD bands, Hjorth)
  05_visualizations.py          # 13 exploratory figures
  06_class_tests.py             # ANOVA / Kruskal / paired Wilcoxon / Cliff's delta + FDR
  07_dimreduction.py            # PCA / t-SNE / UMAP × {raw, global-z, subject-z}
  08_baselines.py               # LOSO × 3 models × 2 preprocs × 3 label schemes
results/
  tables/                       # all CSV + parquet deliverables
  reports/                      # per-stage markdown summaries + FINDINGS.md (this file)
plots/
  inventory/ raw_stats/ physio/ tfdomain/ viz/ class_tests/ dimreduction/ baselines/
cache/                          # primed *.npz / *.parquet so scripts are instant on rerun
```

Everything reruns end-to-end with `uv run python scripts/<n>.py` — no manual steps.
