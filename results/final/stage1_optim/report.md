# Stage 1 Per-Modality Optimization

| Modality | Pool | Best TopK (K) | TopK LOSO AUC | TopK VAL AUC | Best Fwd (K) | Fwd LOSO AUC | Fwd VAL AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BVP | 52 | 30 | 0.8515 | 0.8662 | 24 | 0.8603 | 0.8579 |
| EDA | 46 | 15 | 0.8031 | 0.8304 | 23 | 0.8234 | 0.8291 |
| RESP | 44 | 5 | 0.5491 | 0.5953 | 6 | 0.5910 | 0.5748 |
| SpO2 | 35 | 10 | 0.5857 | 0.6347 | 11 | 0.6366 | 0.6400 |
| EDA+BVP | 98 | 75 | 0.8841 | 0.8997 | 8 | 0.8970 | 0.8957 |
| EDA+BVP+RESP | 142 | 75 | 0.8847 | 0.9003 | 8 | 0.8970 | 0.8957 |
| All | 177 | 100 | 0.8899 | 0.9012 | 8 | 0.8970 | 0.8957 |

## BVP — Top Ranked Features (cliff delta)

```
 1. Bvp_diff_std
 2. Bvp_p90
 3. Bvp_peak_power
 4. Bvp_bp_vhf
 5. Bvp_total_power
 6. Bvp_bp_vhf_rel
 7. Bvp_iqr
 8. Bvp_spec_entropy
 9. Bvp_mad
10. Bvp_hjorth_activity
11. Bvp_std
12. Bvp_cv
13. Bvp_spec_flatness
14. Bvp_max
15. Bvp_mean_freq
16. Bvp_spec_centroid
17. BVP_env_std
18. Bvp_zcr
19. Bvp_mcr
20. BVP_peak_amp_mean
```

## BVP — Greedy Forward Selected

```
 1. Bvp_p90
 2. Bvp_diff_std
 3. Bvp_peak_rel_power
 4. Bvp_max
 5. Bvp_range
 6. Bvp_bp_vhf_rel
 7. Bvp_bp_vhf
 8. Bvp_zcr
 9. Bvp_mcr
10. Bvp_petrosian_fd
11. BVP_peak_amp_mean
12. Bvp_hjorth_mobility
13. Bvp_n_extrema
14. Bvp_hjorth_complexity
15. Bvp_spec_centroid
16. Bvp_mean_freq
17. Bvp_spec_flatness
18. Bvp_dfa_alpha
19. BVP_env_std
20. Bvp_rms
```

## EDA — Top Ranked Features (cliff delta)

```
 1. Eda_mean_abs
 2. Eda_mean
 3. Eda_max
 4. Eda_median
 5. Eda_hjorth_complexity
 6. Eda_energy
 7. Eda_p90
 8. Eda_rms
 9. Eda_hjorth_mobility
10. Eda_dfa_alpha
11. Eda_bp_higher
12. Eda_spec_bandwidth
13. Eda_spec_spread
14. EDA_tonic_mean
15. Eda_p10
16. Eda_mean_freq
17. Eda_spec_centroid
18. Eda_min
19. Eda_range
20. EDA_range
```

## EDA — Greedy Forward Selected

```
 1. Eda_dfa_alpha
 2. Eda_rms
 3. Eda_spec_spread
 4. Eda_range
 5. Eda_spec_centroid
 6. Eda_hjorth_mobility
 7. EDA_scr_count
 8. Eda_spec_bandwidth
 9. Eda_mean_freq
10. EDA_range
11. Eda_spec_flatness
12. Eda_spec_entropy
13. Eda_n_extrema
14. EDA_scr_rate
15. Eda_petrosian_fd
16. EDA_tonic_slope
17. Eda_p90
18. Eda_energy
19. Eda_samp_entropy
20. Eda_bp_higher_rel
```

## RESP — Top Ranked Features (cliff delta)

```
 1. Resp_median
 2. Resp_cv
 3. Resp_peak_rel_power
 4. RESP_rate
 5. Resp_mean
 6. RESP_ie_ratio
 7. Resp_spec_entropy
 8. Resp_energy
 9. Resp_n_extrema
10. Resp_mean_abs
11. Resp_p10
12. Resp_samp_entropy
13. Resp_rms
14. Resp_min
15. Resp_hjorth_complexity
16. Resp_petrosian_fd
17. Resp_diff_std
18. Resp_spec_edge_95
19. RESP_amp_mean
20. Resp_approx_entropy
```

## RESP — Greedy Forward Selected

```
 1. RESP_rate
 2. Resp_dfa_alpha
 3. Resp_spec_edge_95
 4. Resp_peak_freq
 5. Resp_skew
 6. Resp_min
 7. Resp_hjorth_activity
 8. Resp_rms
 9. Resp_n_extrema
10. Resp_iqr
11. Resp_median
12. Resp_mean_abs
13. RESP_ie_ratio
```

## SpO2 — Top Ranked Features (cliff delta)

```
 1. SPO2_std
 2. SpO2_std
 3. SpO2_cv
 4. SpO2_total_variance
 5. SpO2_hjorth_activity
 6. SpO2_peak_power
 7. SpO2_total_power
 8. SPO2_dip_magnitude
 9. SpO2_range
10. SpO2_iqr
11. SpO2_diff_std
12. SpO2_mad
13. SPO2_slope
14. SpO2_p10
15. SpO2_perm_entropy
16. SpO2_petrosian_fd
17. SpO2_rms
18. SpO2_mean
19. SPO2_mean
20. SpO2_mean_abs
```

## SpO2 — Greedy Forward Selected

```
 1. SpO2_peak_power
 2. SpO2_zcr
 3. SpO2_iqr
 4. SpO2_samp_entropy
 5. SpO2_median
 6. SpO2_std
 7. SpO2_diff_std
 8. SpO2_perm_entropy
 9. SpO2_petrosian_fd
10. SpO2_mcr
11. SPO2_std
```

## EDA+BVP — Top Ranked Features (cliff delta)

```
 1. Bvp_diff_std
 2. Bvp_p90
 3. Bvp_peak_power
 4. Bvp_bp_vhf
 5. Bvp_total_power
 6. Bvp_bp_vhf_rel
 7. Bvp_iqr
 8. Bvp_spec_entropy
 9. Bvp_mad
10. Bvp_hjorth_activity
11. Bvp_std
12. Bvp_cv
13. Bvp_spec_flatness
14. Bvp_max
15. Bvp_spec_centroid
16. Bvp_mean_freq
17. Eda_mean
18. Eda_mean_abs
19. Eda_max
20. BVP_env_std
```

## EDA+BVP — Greedy Forward Selected

```
 1. Bvp_p90
 2. Eda_spec_spread
 3. Eda_energy
 4. Eda_hjorth_complexity
 5. Eda_bp_higher
 6. Eda_mean_abs
 7. BVP_env_std
 8. Bvp_bp_vhf_rel
 9. Eda_spec_bandwidth
10. Eda_mean
11. Eda_hjorth_mobility
12. Eda_rms
13. BVP_peak_amp_mean
14. Bvp_spec_flatness
15. Bvp_cv
16. Bvp_std
17. Bvp_hjorth_activity
18. Bvp_zcr
19. Bvp_mcr
20. Bvp_total_power
```

## EDA+BVP+RESP — Top Ranked Features (cliff delta)

```
 1. Bvp_diff_std
 2. Bvp_p90
 3. Bvp_peak_power
 4. Bvp_bp_vhf
 5. Bvp_total_power
 6. Bvp_bp_vhf_rel
 7. Bvp_iqr
 8. Bvp_spec_entropy
 9. Bvp_mad
10. Bvp_hjorth_activity
11. Bvp_std
12. Bvp_cv
13. Bvp_spec_flatness
14. Bvp_max
15. Bvp_spec_centroid
16. Bvp_mean_freq
17. Eda_mean_abs
18. Eda_mean
19. Eda_max
20. BVP_env_std
```

## EDA+BVP+RESP — Greedy Forward Selected

```
 1. Bvp_p90
 2. Eda_spec_spread
 3. Eda_energy
 4. Eda_hjorth_complexity
 5. Eda_bp_higher
 6. Eda_mean_abs
 7. BVP_env_std
 8. Bvp_bp_vhf_rel
 9. Eda_spec_bandwidth
10. Eda_mean
11. Eda_hjorth_mobility
12. Eda_rms
13. BVP_peak_amp_mean
14. Bvp_spec_flatness
15. Bvp_cv
16. Bvp_std
17. Bvp_hjorth_activity
18. Bvp_zcr
19. Bvp_mcr
20. Bvp_total_power
```

## All — Top Ranked Features (cliff delta)

```
 1. Bvp_diff_std
 2. Bvp_p90
 3. Bvp_peak_power
 4. Bvp_bp_vhf
 5. Bvp_total_power
 6. Bvp_bp_vhf_rel
 7. Bvp_iqr
 8. Bvp_spec_entropy
 9. Bvp_mad
10. Bvp_hjorth_activity
11. Bvp_std
12. Bvp_cv
13. Bvp_spec_flatness
14. Bvp_max
15. Bvp_mean_freq
16. Bvp_spec_centroid
17. Eda_mean
18. Eda_mean_abs
19. Eda_max
20. BVP_env_std
```

## All — Greedy Forward Selected

```
 1. Bvp_p90
 2. Eda_spec_spread
 3. Eda_energy
 4. Eda_hjorth_complexity
 5. Eda_bp_higher
 6. Eda_mean_abs
 7. BVP_env_std
 8. Bvp_bp_vhf_rel
 9. Eda_spec_bandwidth
10. Eda_mean
11. Eda_hjorth_mobility
12. Eda_rms
13. BVP_peak_amp_mean
14. Bvp_spec_flatness
15. Bvp_cv
16. Bvp_std
17. Bvp_hjorth_activity
18. Bvp_zcr
19. Bvp_mcr
20. Bvp_total_power
```
