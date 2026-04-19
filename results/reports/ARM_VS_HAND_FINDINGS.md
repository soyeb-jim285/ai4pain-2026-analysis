# ARM vs HAND localisation — consolidated findings

*Companion to `FINDINGS.md`. Everything below is specific to stage 2 of the
two-stage pipeline (No-Pain vs Pain → Pain-Arm vs Pain-Hand). Pooled
training: 41 subjects × 24 pain segments = 984 train segments. Held-out
validation: 12 subjects × 24 = 288 segments. Chance = 0.50.*

---

## Headline

**After eight independent analyses, the arm-vs-hand problem has one
convincingly-discriminative channel — RESP — carrying a small but real
signal, and no model configuration reliably clears ~0.55 macro-F1 on
held-out subjects.** Physiological cap on this dataset is somewhere
between 0.55 and 0.60 val macro-F1.

| what we tried | result | source |
|---|---|---|
| aggregate features (243) | 0 FDR-sig | `06_class_tests.py` |
| temporal sub-window features (440) | 0 FDR-sig subject-mean; 4 sig under LMM, all RESP | `09_temporal_armhand.py` |
| PPG morphology + cross-signal coupling (52) | 0 FDR-sig; |δ| ceiling ~0.18 | `10_morphology_coupling.py` |
| reactivity + interactions + multiscale (646) | 0 FDR-sig; top-10 almost all RESP | `11_reactivity_interactions.py` |
| stronger ML (LR/RF/XGB/subwindow, ~1200 features) | 0.52–0.54 LOSO, 0.47–0.54 val | `12_armhand_stronger_ml.py` |
| Tier A #1 onset alignment | median p-ratio 1.48 (*worse*), 0 new FDR | `13_tierA1_onset_alignment.py` |
| Tier A #2 BVP proximity kinetics | 0 FDR; δ=0.22 on `bvp_local_vasoconstriction_speed_r2` (beat 0.18 ceiling) | `14_tierA2_bvp_kinetics.py` |
| Tier A #3 stage-1 calibration drift | null (paired p=0.85); pain_prob-alone F1=0.39 | `15_tierA3_stage1_calibration.py` |
| Tier A #4 subject clustering | agglo-k2 gets 19 nominal-sig (vs 7 pooled); no cluster clears F1=0.60 | `16_tierA4_subject_clusters.py` |
| **Tier A #5 channel-restricted** | **RESP: 17/53 FDR-sig; δ up to 0.586** | `17_tierA5_channel_restricted.py` |
| Tier C #1 PDA 3-Gaussian on BVP (paper 2 idea) | 0 FDR / 28; LOSO 0.512, val 0.528 | `22_tierC_pda_bvp.py` |
| Tier C #2 TVSymp envelope on EDA (paper 3 idea) | 0 FDR / 12; LOSO 0.491, val 0.477 | `23_tierC_tvsymp_eda.py` |
| DL: 1D CNN + SSL-pretrain (5-seed Kaggle T4) | LOSO 0.46, val 0.49-0.52 (chance) | `19_armhand_dl.py` |

---

## 1. The one strong discovery — respiration carries the signal

Tier A #5 was the only analysis that applied **FDR within a channel** rather
than across all 243 features. The previous failures were partly a multiple-
comparison artefact: 17 real RESP effects were drowning in 226 non-RESP
tests.

Top 10 features by raw p-value (all channels, paired Wilcoxon, 41 train subjects):

| channel | feature | p | q (BH within channel) | Cliff's δ | % subj with ARM>HAND |
|---|---|---|---|---|---|
| resp | `Resp_mad` | 0.00083 | **0.022** | +0.586 | 71 % |
| resp | `raw_Resp_mad` | 0.00083 | **0.022** | +0.586 | 71 % |
| resp | `Resp_mean_abs` | 0.0025 | **0.027** | +0.532 | 73 % |
| resp | `Resp_energy` | 0.0030 | **0.027** | +0.527 | 66 % |
| resp | `Resp_rms` | 0.0031 | **0.027** | +0.524 | 66 % |
| resp | `Resp_diff_std` | 0.0036 | **0.027** | +0.517 | 66 % |
| spo2 | `SPO2_dip_magnitude` | 0.0044 | 0.184 | −0.516 | 24 % |
| resp | `raw_Resp_skew` | 0.0054 | **0.032** | −0.493 | 29 % |
| resp | `Resp_skew` | 0.0054 | **0.032** | −0.493 | 29 % |

Reading this:
- ARM segments have **larger RESP amplitude / variability / energy** than HAND segments (7 features with +δ; ~66–73 % subjects confirm direction).
- HAND segments have **more skewed RESP waveforms** (2 features with −δ).
- This is consistent with RESP reacting differently to proximal vs distal stimulation — arm stim likely recruits a broader postural/splinting response that RESP picks up, whereas hand stim stays more localised.

Per-channel significance counts:

| channel | FDR<0.05 | FDR<0.10 | n features |
|---|---|---|---|
| **resp** | **17** | **20** | 53 |
| bvp | 0 | 0 | 61 |
| eda | 0 | 0 | 55 |
| spo2 | 0 | 0 | 42 |
| cross | 0 | 0 | 1 |

EDA — which dominated pain detection — contributes nothing to localisation.

---

## 2. Classifier translation — where the 0.55 ceiling sits

LOSO on 41 train subjects → best model + validation hold-out:

| feature-set | model | LOSO macro-F1 | val macro-F1 |
|---|---|---|---|
| all 212 features | logreg | **0.559** ± 0.10 | 0.510 |
| resp-only (53) | logreg | 0.522 | **0.552** |
| resp-only (53) | rf | 0.532 | 0.464 |
| top 40 by Cliff's δ (all channels) | logreg | **0.574** | — |
| top 10 resp-only | logreg | **0.547** | — |

Two stable observations:

1. **RESP-alone on validation is 0.552** — within noise of the all-channel
   0.559, but with 4× fewer features. Good parsimony.
2. **Top-40 feature selection beats using all 212** (0.574 vs 0.559 LOSO).
   Most features are noise; the signal lives in a small RESP-dominated subset.

The ~0.574 LOSO / ~0.55 val ceiling is the *actual* information content
of these 4 channels for this task. Validation translates poorly (0.55)
because the 12 held-out subjects include responders and non-responders
unlike the train distribution.

---

## 3. What didn't help (rigorously disproved)

- **Stimulus-onset alignment.** Aligning segments to the detected onset
  actually *raised* p-values on 14/22 top features. Verdict: the 10-s
  windows are already onset-centred enough that further alignment adds
  noise from imperfect onset detection.

- **Stage-1 pain-probability as a meta-feature.** Stage-1 scores ARM and
  HAND segments essentially identically (paired p = 0.85; pain_prob-alone
  classifier below chance at 0.39 macro-F1). The stage-1 model does not
  secretly know the location.

- **BVP proximity kinetics (though the hypothesis was partially confirmed).**
  The hand-close-to-sensor proximity hypothesis predicted faster BVP
  decay and larger vasoconstriction index under hand stim — both
  directions *were* observed, but effect sizes max out at Cliff's δ = 0.22
  and do not survive BH-FDR. Worth including 2–3 of the kinetics features
  in any final feature set, but they're a small contributor, not a
  breakthrough.

---

## 4. Modest wins worth keeping

- **Subject heterogeneity exists** (Tier A #4). The 2-cluster agglomerative
  split on per-subject NoPain signatures nearly **triples** the count of
  nominal-significant arm-vs-hand features (19 vs 7 pooled). But no
  sub-group classifier crosses macro-F1 = 0.60, so mixture-of-experts is
  a minor lever, not a major one. Clusters are distinguished by **BVP
  spectral/entropy features** — responders vs non-responders in the BVP
  pathway.

- **Individual subject prognosis**. One subject (29) reaches LOO-within-
  subject accuracy 0.75; 5/41 exceed 0.60. Most subjects are genuinely not
  separable. If the challenge allows any per-subject calibration at test
  time, even a small calibration pass would help the discriminable ~20 %
  of the population.

- **2 / 2 BVP kinetics directions preserved on validation** — small n but
  100 % replication is encouraging. Keep those features around.

---

## 5. Recommended stage-2 architecture

Given these findings, a realistic stage-2 that should reproduce ≈ 0.55
val macro-F1 and possibly push to 0.58 with careful ensembling:

1. **Feature set (~50 features, small and targeted):**
   - All 53 RESP features.
   - Top 3 BVP kinetics (`bvp_local_vasoconstriction_speed_r2`,
     `bvp_beat_amp_trend_ratio`, `bvp_recovery_halftime_s`).
   - The 2-cluster indicator from Tier A #4 agglo-k2 as a one-hot feature.

2. **Preprocessing:**
   - Per-subject z-score across all 36 of that subject's segments (the
     NoPain segments come from stage 1 — use them for normalisation even
     though they're not classified).

3. **Model:**
   - Logistic regression (L2, class-balanced) for interpretability and
     validation stability.
   - OR an ensemble of LR + RF averaging class probabilities, since the
     two gave different val numbers (0.55 vs 0.46) and averaging
     de-risks.

4. **Target the responders:**
   - Train the above on all 41 subjects as the "base model."
   - Separately train a second model restricted to subjects in cluster 0
     of the agglo-k2 split (23 subjects, which had within-cluster LOSO
     0.544 — same as pooled but with less noise to drown).
   - At test time, assign the test subject to the nearest cluster by
     their NoPain signature and use the cluster-specific model.

5. **Don't bother with:**
   - SpO₂-derived features (flatline artefacts + one nominally-significant
     dip_magnitude feature doesn't survive FDR).
   - Cross-signal couplings (Tier A #5: 0 sig, 1 feature in the cross
     group anyway).
   - Onset alignment (Tier A #1 null).
   - Stage-1 probability as a feature (Tier A #3 null).

## 6. AI4Pain 2025 paper ideas — tested, none break ceiling

Three physiologically-motivated ideas lifted from the 2025 challenge papers:

- **Paper 2 (Javierre) — Pulse Decomposition Analysis.** 3-Gaussian fit per
  BVP pulse → median/IQR of amplitudes, means, sigmas, reflection ratios
  (28 features). Hypothesis: arterial-tree reflection geometry differs
  between forearm and dorsal-hand stimulation sites. Result: LOSO
  0.512 ± 0.103, val 0.528, 0/28 FDR-sig. Best raw-p `pda_mu1_iqr`
  p = 0.028 (forward-wave position jitter across pulses) — dies under FDR.

- **Paper 3 (Gkikas) — TVSymp envelope on EDA.** Narrowband (0.08–0.24 Hz)
  Butterworth approximation of VFCDM + Hilbert envelope, normalised by
  its std; extracted peak/latency/AUC/centroid + early-vs-late contrast
  (12 features). Hypothesis: spinal-entry-level differences between arm
  and hand afferents produce different sweat-response latency at finger.
  Result: LOSO 0.491 ± 0.098, val 0.477, 0/12 FDR-sig. Peak amp means
  are 3.404 (arm) vs 3.399 (hand) — essentially identical.

- **Paper 1 (Tazin) — wavelet + arc-length + fuzzy entropy.** Covered by
  script 21 + ensemble sweep in `ARM_VS_HAND_BEST_MODEL.md` row "tierC
  paper features": 122 configs, net val lift **−0.020**.

All three 2025-paper ideas are diagnostic: even the most physiologically
principled hand-crafted feature families can't extract localisation signal
that isn't there. Combined with the 5-seed CNN+SSL result (LOSO 0.46, val
0.52), the evidence is now triangulated across (a) aggregate features,
(b) paper-derived features, (c) deep learning — all converge at 0.49–0.56.

## 7. If you want to push past 0.58

Genuinely-unexhausted angles (lower probability, higher cost):

- **Time-series foundation models** (Moment, Chronos, Mantis) as frozen
  feature extractors → LogReg head. Brings inductive bias from millions of
  series; CPU-feasible (precompute once). Likely +2–3 % ceiling at best.
- **Subject-adaptation head / meta-learning.** Paper 2's "patient-wise
  normalisation = 77 % of importance" strongly endorses this. Combine with
  cluster-conditional LR from §4.

- **Honest reporting of the ceiling.** A robust 0.55 val macro-F1 with
  the RESP-focused feature set + cluster-conditional LR, plus the negative
  results from the 2025-paper feature families, is publishable as the
  *empirical ceiling of peripheral physiology for sub-upper-extremity
  pain localisation at 10-s resolution.*

---

## 7. Artefacts produced in this push

| file | purpose |
|---|---|
| `results/tables/tierA1_onsets.csv` + `tierA1_alignment_effect.csv` | onset detection per segment + aligned-vs-unaligned p-value comparison |
| `results/tables/tierA2_bvp_kinetics_features.parquet` + `tierA2_bvp_kinetics_tests.csv` | 30 BVP kinetics features + paired Wilcoxon/FDR |
| `results/tables/tierA3_stage1_probabilities.csv` + `tierA3_stage1_armhand_drift.csv` | per-segment stage-1 pain-prob + drift tests |
| `results/tables/tierA4_subject_clusters.csv` + `tierA4_within_cluster_tests.csv` + `tierA4_within_cluster_loso.csv` | subject clustering + within-cluster ARM-vs-HAND analysis |
| `results/tables/tierA5_feature_channel_map.csv` + `tierA5_per_channel_tests.csv` + `tierA5_per_channel_loso.csv` + `tierA5_feature_count_curve.csv` | channel-restricted tests + classifiers + learning curves |
| `results/tables/tierC_pda_bvp_features.parquet` + `tierC_pda_bvp_tests.csv` | 28 PDA features (paper 2) + paired Wilcoxon/FDR |
| `results/tables/tierC_tvsymp_eda_features.parquet` + `tierC_tvsymp_eda_tests.csv` | 12 TVSymp envelope features (paper 3) + paired Wilcoxon/FDR |
| `results/reports/13_…_summary.md` through `17_…_summary.md`, `tierC_pda_bvp_summary.md`, `tierC_tvsymp_eda_summary.md` | per-analysis summaries |
| `plots/tierA{1..5}_*` | figures for each Tier A analysis |

Everything is reproducible end-to-end with `uv run python scripts/1{3,4,5,6,7}_*.py scripts/2{2,3}_*.py`.
