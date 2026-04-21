# Stage 2 Sub-Window Vote / Concat

- n_sub: 5  signals: ['Resp']  calibration: isotonic
- hard-10 set (from combined_best): [4, 11, 15, 22, 25, 45, 46, 55, 56, 63]

| variant   |   n_sub |   loso_macro_f1 |   loso_per_subject_std |   loso_per_subject_min |   val_macro_f1 |   val_per_subject_std |   val_per_subject_min |   hard10_loso_macro_f1 |
|:----------|--------:|----------------:|-----------------------:|-----------------------:|---------------:|----------------------:|----------------------:|-----------------------:|
| vote      |       5 |          0.5427 |                 0.1236 |                 0.3333 |         0.4375 |                0.0970 |                0.2500 |                 0.5000 |
| concat    |       5 |          0.5549 |                 0.0894 |                 0.4167 |         0.4861 |                0.1310 |                0.2500 |                 0.5333 |