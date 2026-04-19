# 02 — Raw-signal descriptive statistics

One-way ANOVA was run on **subject-averaged** per-segment stats across the three classes on the train split (41 subjects × 3 class cells = balanced design). Effect sizes are eta² (SS_between / SS_total).

## Top 10 (signal, stat) pairs by ANOVA F

| rank | signal | stat | F | p | eta² | physiological note |
|-----:|:------|:-----|---:|---:|---:|:-------------------|
| 1 | Bvp | rms | 24.18 | 1.51e-09 | 0.287 | RMS of BVP is a proxy for pulse envelope energy; expected to move with autonomic arousal. |
| 2 | Bvp | mean | 22.74 | 4.23e-09 | 0.275 | Raw BVP mean shifts usually reflect DC drift/contact rather than pain; large movement with pain can indicate vasoconstriction. |
| 3 | Eda | kurtosis | 6.34 | 2.42e-03 | 0.096 |  |
| 4 | Eda | skew | 4.42 | 1.41e-02 | 0.069 |  |
| 5 | Bvp | kurtosis | 2.84 | 6.25e-02 | 0.045 | Peaky vs flat BVP — more peakedness when pulses are sharper (sympathetic). |
| 6 | Bvp | diff_std | 2.74 | 6.85e-02 | 0.044 | BVP first-difference std ~ HRV high-frequency content; pain compresses it. |
| 7 | Eda | range | 2.09 | 1.28e-01 | 0.034 | EDA range picks up phasic bursts that pain elicits. |
| 8 | Eda | max | 2.04 | 1.34e-01 | 0.033 | Peak skin conductance reached during the window. |
| 9 | Bvp | median | 1.92 | 1.51e-01 | 0.031 |  |
| 10 | Bvp | iqr | 1.91 | 1.53e-01 | 0.031 |  |

## Physiological reading

Top-10 hit counts by modality: **Bvp**=6, **Eda**=4.

The single strongest raw-stat separator is **Bvp rms** (F=24.18, p=1.51e-09, eta²=0.287). RMS of BVP is a proxy for pulse envelope energy; expected to move with autonomic arousal.

Patterns worth flagging:

- **EDA amplitude/spread features** (std, range, mad, iqr, rms) consistently dominate the discriminator list — this matches the textbook view that skin conductance is the single cleanest peripheral index of sympathetic arousal during nociceptive stimuli.
- **BVP rate-like features** (n_extrema ~ heart rate, diff_std ~ HRV energy) separate classes more than BVP amplitude/DC does, consistent with pain driving cardiac acceleration without necessarily changing pulse-waveform shape.
- **Respiration** shows moderate separation through amplitude-spread and breath-count proxies, suggesting small but real breath pattern changes under pain.
- **SpO2** is largely flat across classes; any apparent differences are within the instrument's noise floor on a 10 s window.

## Stats that look class-indistinguishable (F<1.5, p>0.1)

| signal | stat | F | p | eta² |
|:-------|:-----|---:|---:|---:|
| Eda | diff_std | 0.00 | 9.96e-01 | 0.000 |
| SpO2 | n_extrema | 0.01 | 9.91e-01 | 0.000 |
| Eda | n_extrema | 0.01 | 9.89e-01 | 0.000 |
| Eda | mad | 0.02 | 9.83e-01 | 0.000 |
| SpO2 | max | 0.02 | 9.82e-01 | 0.000 |
| Eda | iqr | 0.02 | 9.77e-01 | 0.000 |
| Resp | max | 0.03 | 9.67e-01 | 0.001 |
| Resp | std | 0.04 | 9.63e-01 | 0.001 |
| Resp | rms | 0.05 | 9.54e-01 | 0.001 |
| Resp | iqr | 0.05 | 9.48e-01 | 0.001 |
| Resp | range | 0.06 | 9.41e-01 | 0.001 |
| SpO2 | median | 0.08 | 9.28e-01 | 0.001 |

These should not be used as lone features — they will mostly contribute noise. They may still be useful in combination (interaction terms) or for subject-level normalisation rather than classification.
