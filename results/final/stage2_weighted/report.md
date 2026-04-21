# Stage 2 Weighted Training Sweep

- feature_set: `resp_all`  scaler: `robust`  cal: `isotonic`
- norm: subject_z (fixed)
- k_hard: 10 (bottom by reactivity)

## Results

| mode           |   loso_macro_f1 |   loso_per_subject_std |   loso_per_subject_min |   val_macro_f1 |   val_per_subject_std |   val_per_subject_min |   loso_hard10_macro_f1 |   loso_easy10_macro_f1 |
|:---------------|----------------:|-----------------------:|-----------------------:|---------------:|----------------------:|----------------------:|-----------------------:|-----------------------:|
| uniform        |          0.5224 |                 0.0974 |                 0.3333 |         0.5139 |                0.0889 |                0.3333 |                 0.5333 |                 0.5000 |
| squared_inv    |          0.5224 |                 0.0957 |                 0.2500 |         0.5625 |                0.0691 |                0.4167 |                 0.4917 |                 0.5167 |
| equal_subject  |          0.5224 |                 0.0974 |                 0.3333 |         0.5139 |                0.0889 |                0.3333 |                 0.5333 |                 0.5000 |
| inv_reactivity |          0.5163 |                 0.0924 |                 0.2500 |         0.5347 |                0.0718 |                0.4167 |                 0.5083 |                 0.5333 |
| boost_hard     |          0.5163 |                 0.1108 |                 0.2500 |         0.5556 |                0.0856 |                0.4167 |                 0.5000 |                 0.5167 |