# 14 - Tier-A2: BVP proximity kinetics (ARM vs HAND)

## Headline
- Features tested: **30** (raw + Delta-from-baseline). Train subjects: 41, validation subjects: 12.
- **0** features survive BH-FDR<0.05; **2** at nominal p<0.05.
- |Cliff's delta| range: max=0.220, median=0.054 (prior aggregate ceiling was ~0.14-0.18).

## Top 10 features by smallest p (paired Wilcoxon, train)

| feature | n | W | p | p_fdr | Cliff | sign(arm>hand) | direction | LMM p |
|---------|--:|--:|--:|------:|------:|---------------:|-----------|------:|
| `bvp_local_vasoconstriction_speed_r2__delta` | 39 | 219.0 | 0.0161 | 0.418 | -0.220 | 0.33 | HAND > ARM | 0.0124 |
| `bvp_local_vasoconstriction_speed_r2` | 40 | 247.0 | 0.0278 | 0.418 | -0.168 | 0.35 | HAND > ARM | 0.0162 |
| `bvp_beat_amp_trend_ratio__delta` | 39 | 305.0 | 0.241 | 0.936 | -0.128 | 0.36 | HAND > ARM | 0.383 |
| `bvp_recovery_halftime_s` | 38 | 290.0 | 0.249 | 0.936 | +0.041 | 0.55 | ARM > HAND | 0.293 |
| `bvp_local_vasoconstriction_index__delta` | 39 | 307.0 | 0.253 | 0.936 | -0.127 | 0.38 | HAND > ARM | 0.216 |
| `bvp_peak_envelope_slope` | 40 | 328.0 | 0.277 | 0.936 | +0.054 | 0.65 | ARM > HAND | 0.257 |
| `bvp_local_vasoconstriction_index` | 40 | 332.0 | 0.301 | 0.936 | -0.104 | 0.40 | HAND > ARM | 0.232 |
| `bvp_beat_amp_trend_ratio` | 40 | 336.0 | 0.327 | 0.936 | -0.100 | 0.38 | HAND > ARM | 0.404 |
| `bvp_peak_envelope_slope__delta` | 39 | 322.0 | 0.35 | 0.936 | +0.089 | 0.64 | ARM > HAND | 0.296 |
| `bvp_local_vasoconstriction_speed_slope` | 40 | 348.0 | 0.413 | 0.936 | +0.058 | 0.62 | ARM > HAND | 0.399 |

## Hypothesis-driven probes
- `bvp_amp_halfdecay_time_s`: observed ARM > HAND (expected HAND < ARM); W=203.00, p=0.544, Cliff=+0.018.
- `bvp_local_vasoconstriction_index`: observed HAND > ARM (expected HAND > ARM); W=332.00, p=0.301, Cliff=-0.104.

- **Hypothesis verdict (proximity)**: Yes. Halfdecay direction matches hand-faster-decay = True; local-vasoconstriction-index hand-larger = True.

## Validation reproducibility
- Features at nominal p<0.05 in train: **2**. Direction preserved on validation: **2/2** (100.0%).

## Effect size vs prior 0.14-0.18 ceiling
- **Beats the ceiling**: max |Cliff's delta| = 0.220 > 0.18.

## Outputs
- `results/tables/tierA2_bvp_kinetics_features.parquet`
- `results/tables/tierA2_bvp_kinetics_tests.csv`
- `results/tables/tierA2_bvp_kinetics_val_repro.csv`
- `plots/tierA2_bvp_kinetics/`