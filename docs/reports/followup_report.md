# Follow-up study: split sensitivity, extended searches and controls

Built by `src/followup_report.py` from `runs/` and `splits/session_blocked/folds.csv`; do not edit by hand.
Study plan: handbook §24, rows of 2026-11-14. All scores are percentages.

**Which photos each number comes from**

- Sections 3 and 7 are **validation** scores (used for selection only).
- Sections 4 and 5 are **test** scores of the **locked split** (`splits/folds.csv`).
- Section 6 is **test** scores of the **session-blocked split** (`splits/session_blocked/folds.csv`).
- Sections 1 and 2 show both splits side by side (test).
- "mean ± SD" is over the five rounds. Both ranges are around the pooled score (every test photo once), so a
  mean can sit slightly outside its range when the rounds differ a lot.
- "photo range" is the declared 95% photo bootstrap. "block range" resamples whole capture blocks (photos
  shot without a pause of more than 60 s from one camera) and is wider when neighbouring photos are alike.

**Limits to read first**

- The wider searches (edge rule, second aggregator study) and the same-head whole-photo control were declared
  **after** the core test results were seen (2026-11-14). Their selections use validation scores only.
- The session split retrains the nine core configurations with the settings chosen on locked-split validation
  photos. Some of those photos are session-split test photos, so these scores are a sensitivity analysis, not a
  clean new tuning round.
- Class shares on the session split are reported, not enforced (`docs/reports/session_split_report.md`).
- Tuning budgets are not equal in section 5: each aggregator keeps the better of two 40-trial studies, the
  whole-photo controls had one 40-trial study, and a CNN may have up to 3 added learning rates.
  Differences in section 5 therefore include a difference in search effort.
- The photo range is stratified by class; the block range resamples whole blocks and is not, so it also
  contains class-count variation. That is part of why it is wider.

## 1. Shortcut controls (test, mean ± SD over 5 rounds)

| baseline | locked macro-F1 | locked balanced acc. | session macro-F1 | session balanced acc. |
|---|---|---|---|---|
| majority | 15.4 ± 0.0 | 25.0 ± 0.0 | 15.4 ± 1.1 | 25.0 ± 0.0 |
| random_prior | 25.2 ± 2.1 | 25.2 ± 2.1 | 25.2 ± 1.6 | 25.6 ± 1.7 |
| colour_only | 52.9 ± 2.5 | 55.0 ± 2.6 | 43.5 ± 2.8 | 45.2 ± 3.6 |
| background_only | 48.9 ± 1.6 | 51.8 ± 1.6 | 39.7 ± 3.6 | 41.8 ± 4.3 |
| properties_only | 32.3 ± 2.8 | 36.1 ± 3.2 | 29.5 ± 3.1 | 33.1 ± 3.5 |
| camera_only | 17.4 ± 1.5 | 30.9 ± 1.0 | 15.1 ± 5.1 | 30.5 ± 4.2 |
| time_neighbour | 80.5 ± 1.6 | 80.5 ± 1.6 | 30.4 ± 10.8 | 33.7 ± 12.1 |
| background_strict | 45.8 ± 2.8 | 49.0 ± 2.4 | 38.9 ± 4.9 | 43.3 ± 5.4 |

## 2. Locked split against session-blocked split (test macro-F1, nine core configurations)

| method | locked macro-F1 (mean ± SD) | locked macro-F1 photo range | locked macro-F1 block range | session macro-F1 (mean ± SD) | session macro-F1 photo range | session macro-F1 block range | drop (points) |
|---|---|---|---|---|---|---|---|
| inceptionv3_224 | 80.9 ± 2.7 | [79.5, 82.3] | [76.0, 83.4] | 72.7 ± 4.3 | [72.7, 75.8] | [68.7, 76.7] | 8.3 |
| efficientnetv2s_224 | 82.8 ± 1.0 | [81.4, 84.1] | [78.4, 85.0] | 71.7 ± 7.4 | [71.8, 74.9] | [66.4, 77.1] | 11.0 |
| convnext_tiny_224 | 83.4 ± 2.6 | [82.1, 84.7] | [79.3, 85.5] | 75.0 ± 5.0 | [74.4, 77.4] | [70.5, 79.1] | 8.4 |
| swin_tiny_224 | 83.7 ± 1.3 | [82.5, 85.0] | [80.0, 85.7] | 75.9 ± 3.3 | [75.5, 78.4] | [71.9, 79.3] | 7.8 |
| probe_224 | 78.2 ± 0.7 | [76.8, 79.7] | [73.3, 80.5] | 72.0 ± 4.6 | [71.7, 74.7] | [67.7, 75.4] | 6.2 |
| probe_512 | 81.3 ± 1.1 | [79.9, 82.6] | [76.5, 83.3] | 73.5 ± 5.9 | [74.0, 77.1] | [69.2, 78.4] | 7.8 |
| mean | 76.0 ± 1.5 | [74.5, 77.6] | [70.7, 78.4] | 66.2 ± 8.7 | [66.8, 70.3] | [60.2, 72.8] | 9.8 |
| max | 82.9 ± 1.7 | [81.5, 84.2] | [79.2, 84.6] | 73.9 ± 6.7 | [74.0, 77.1] | [69.8, 78.4] | 9.0 |
| attention | 82.4 ± 2.0 | [81.0, 83.6] | [78.2, 84.5] | 75.7 ± 4.6 | [75.3, 78.2] | [71.8, 79.5] | 6.7 |

## 3. Searches (validation macro-F1, mean over 5 rounds)

| method | value tried | mean val macro-F1 | in declared grid / study | selected | selection changed |
|---|---|---|---|---|---|
| inceptionv3_224 | 3e-05 | 73.3 | yes |  | no |
| inceptionv3_224 | 0.0001 | 80.2 | yes |  | no |
| inceptionv3_224 | 0.0003 | 83.1 | yes | yes | no |
| inceptionv3_224 | 0.001 | 81.0 | added |  | no |
| efficientnetv2s_224 | 3e-05 | 74.3 | yes |  | yes |
| efficientnetv2s_224 | 0.0001 | 83.4 | yes |  | yes |
| efficientnetv2s_224 | 0.0003 | 84.7 | yes |  | yes |
| efficientnetv2s_224 | 0.001 | 85.9 | added | yes | yes |
| efficientnetv2s_224 | 0.003 | 84.6 | added |  | yes |
| convnext_tiny_224 | 3e-05 | 83.9 | yes |  | no |
| convnext_tiny_224 | 0.0001 | 85.5 | yes |  | no |
| convnext_tiny_224 | 0.0003 | 86.1 | yes | yes | no |
| convnext_tiny_224 | 0.001 | 84.8 | added |  | no |
| swin_tiny_224 | 1e-05 | 82.6 | added |  | no |
| swin_tiny_224 | 3e-05 | 86.5 | yes | yes | no |
| swin_tiny_224 | 0.0001 | 86.2 | yes |  | no |
| swin_tiny_224 | 0.0003 | 84.3 | yes |  | no |
| probe_224 | 0.001 | 73.4 | yes |  | no |
| probe_224 | 0.01 | 72.9 | yes |  | no |
| probe_224 | 0.1 | 72.8 | yes |  | no |
| probe_224 | 1 | 74.9 | yes |  | no |
| probe_224 | 10 | 75.7 | yes |  | no |
| probe_224 | 100 | 77.6 | yes | yes | no |
| probe_224 | 1000 | 76.9 | added |  | no |
| probe_512 | 0.001 | 76.5 | yes |  | no |
| probe_512 | 0.01 | 76.2 | yes |  | no |
| probe_512 | 0.1 | 77.3 | yes |  | no |
| probe_512 | 1 | 78.5 | yes |  | no |
| probe_512 | 10 | 79.9 | yes |  | no |
| probe_512 | 100 | 80.2 | yes | yes | no |
| probe_512 | 1000 | 78.6 | added |  | no |
| mean | declared | 78.5 | yes |  | yes |
| mean | wide | 80.8 | added | yes | yes |
| max | declared | 84.8 | yes | yes | no |
| max | wide | 84.6 | added |  | no |
| attention | declared | 84.7 | yes |  | yes |
| attention | wide | 84.9 | added | yes | yes |

## 4. Final selections on the locked split (test)

Core runs where the selection is unchanged, new runs where it changed; plus the two whole-photo controls.

| method | macro-F1 (mean ± SD) | macro-F1 photo range | macro-F1 block range | balanced acc. (mean ± SD) | balanced acc. photo range | balanced acc. block range | ECE |
|---|---|---|---|---|---|---|---|
| inceptionv3_224 | 80.9 ± 2.7 | [79.5, 82.3] | [76.0, 83.4] | 81.7 ± 2.2 | [80.4, 83.2] | [77.2, 84.0] | 0.049 |
| efficientnetv2s_224 | 83.5 ± 1.8 | [82.2, 84.9] | [78.9, 86.1] | 83.6 ± 2.0 | [82.2, 85.0] | [79.0, 86.0] | 0.042 |
| convnext_tiny_224 | 83.4 ± 2.6 | [82.1, 84.7] | [79.3, 85.5] | 83.5 ± 2.9 | [82.2, 85.0] | [79.5, 85.5] | 0.052 |
| swin_tiny_224 | 83.7 ± 1.3 | [82.5, 85.0] | [80.0, 85.7] | 84.5 ± 1.9 | [83.2, 85.8] | [80.5, 86.3] | 0.060 |
| probe_224 | 78.2 ± 0.7 | [76.8, 79.7] | [73.3, 80.5] | 79.0 ± 0.5 | [77.4, 80.5] | [73.9, 81.1] | 0.024 |
| probe_512 | 81.3 ± 1.1 | [79.9, 82.6] | [76.5, 83.3] | 82.0 ± 1.1 | [80.6, 83.4] | [76.9, 84.0] | 0.020 |
| mean | 78.5 ± 2.7 | [77.0, 79.9] | [73.1, 81.0] | 78.9 ± 2.5 | [77.3, 80.4] | [72.5, 81.4] | 0.116 |
| max | 82.9 ± 1.7 | [81.5, 84.2] | [79.2, 84.6] | 83.4 ± 1.7 | [82.1, 84.8] | [79.6, 85.0] | 0.122 |
| attention | 82.8 ± 1.5 | [81.5, 84.2] | [78.7, 84.8] | 83.4 ± 1.7 | [82.0, 84.8] | [79.1, 85.2] | 0.075 |
| whole224 | 80.2 ± 1.8 | [78.8, 81.6] | [75.5, 82.4] | 80.4 ± 2.0 | [78.9, 81.8] | [75.3, 82.6] | 0.085 |
| whole512 | 81.8 ± 1.8 | [80.5, 83.1] | [77.1, 84.0] | 81.5 ± 1.9 | [80.1, 82.9] | [76.0, 83.9] | 0.076 |

## 5. Attention against every other final method (locked split, test)

| A (attention) vs | metric | difference (points) | photo range | block range | rounds won | verdict (photo range) | verdict (block range) |
|---|---|---|---|---|---|---|---|
| inceptionv3_224 | macro_f1 | 2.0 | [0.4, 3.4] | [0.1, 4.2] | 2/5 | no clear difference | no clear difference |
| inceptionv3_224 | balanced_accuracy | 1.6 | [0.1, 3.1] | [-0.3, 3.7] | 2/5 | no clear difference | no clear difference |
| efficientnetv2s_224 | macro_f1 | -0.7 | [-2.3, 0.8] | [-2.8, 1.5] | 1/5 | no clear difference | no clear difference |
| efficientnetv2s_224 | balanced_accuracy | -0.3 | [-1.8, 1.3] | [-2.3, 1.9] | 2/5 | no clear difference | no clear difference |
| convnext_tiny_224 | macro_f1 | -0.5 | [-2.1, 1.0] | [-2.5, 1.3] | 3/5 | no clear difference | no clear difference |
| convnext_tiny_224 | balanced_accuracy | -0.2 | [-1.6, 1.4] | [-2.1, 1.7] | 3/5 | no clear difference | no clear difference |
| swin_tiny_224 | macro_f1 | -0.9 | [-2.4, 0.5] | [-2.8, 0.7] | 1/5 | no clear difference | no clear difference |
| swin_tiny_224 | balanced_accuracy | -1.1 | [-2.5, 0.4] | [-2.8, 0.5] | 1/5 | no clear difference | no clear difference |
| probe_224 | macro_f1 | 4.6 | [2.9, 6.2] | [2.8, 7.0] | 5/5 | A beats B | A beats B |
| probe_224 | balanced_accuracy | 4.4 | [2.7, 6.1] | [2.5, 6.8] | 5/5 | A beats B | A beats B |
| probe_512 | macro_f1 | 1.6 | [0.1, 3.1] | [-0.0, 3.7] | 4/5 | A beats B | no clear difference |
| probe_512 | balanced_accuracy | 1.4 | [-0.1, 2.9] | [-0.2, 3.8] | 4/5 | no clear difference | no clear difference |
| mean | macro_f1 | 4.4 | [3.0, 5.7] | [2.9, 6.6] | 5/5 | A beats B | A beats B |
| mean | balanced_accuracy | 4.5 | [3.1, 5.9] | [3.0, 7.4] | 5/5 | A beats B | A beats B |
| max | macro_f1 | -0.0 | [-1.2, 1.2] | [-1.2, 0.9] | 2/5 | no clear difference | no clear difference |
| max | balanced_accuracy | -0.0 | [-1.2, 1.2] | [-1.3, 1.1] | 1/5 | no clear difference | no clear difference |
| whole224 | macro_f1 | 2.7 | [1.0, 4.2] | [0.8, 4.9] | 4/5 | A beats B | A beats B |
| whole224 | balanced_accuracy | 3.0 | [1.3, 4.5] | [1.0, 5.6] | 5/5 | A beats B | A beats B |
| whole512 | macro_f1 | 1.1 | [-0.4, 2.6] | [-0.6, 3.0] | 3/5 | no clear difference | no clear difference |
| whole512 | balanced_accuracy | 1.9 | [0.4, 3.3] | [0.1, 4.7] | 5/5 | A beats B | A beats B |

The verdict differs between the two ranges for: probe_512 (macro_f1): photo range 'A beats B', block range 'no clear difference'.

## 6. Attention against the other eight configurations (session-blocked split, test)

| A (attention) vs | metric | difference (points) | photo range | block range | rounds won | verdict (photo range) | verdict (block range) |
|---|---|---|---|---|---|---|---|
| inceptionv3_224 | macro_f1 | 2.6 | [0.9, 4.2] | [0.5, 5.2] | 5/5 | A beats B | A beats B |
| inceptionv3_224 | balanced_accuracy | 2.7 | [0.9, 4.3] | [0.4, 5.3] | 5/5 | A beats B | A beats B |
| efficientnetv2s_224 | macro_f1 | 3.5 | [1.9, 5.1] | [0.9, 6.8] | 5/5 | A beats B | A beats B |
| efficientnetv2s_224 | balanced_accuracy | 3.1 | [1.5, 4.8] | [0.4, 6.5] | 5/5 | A beats B | A beats B |
| convnext_tiny_224 | macro_f1 | 0.9 | [-0.8, 2.5] | [-1.7, 3.8] | 4/5 | no clear difference | no clear difference |
| convnext_tiny_224 | balanced_accuracy | 0.5 | [-1.2, 2.1] | [-2.0, 3.2] | 3/5 | no clear difference | no clear difference |
| swin_tiny_224 | macro_f1 | -0.1 | [-1.7, 1.5] | [-2.2, 2.0] | 2/5 | no clear difference | no clear difference |
| swin_tiny_224 | balanced_accuracy | -0.1 | [-1.8, 1.5] | [-2.1, 2.0] | 3/5 | no clear difference | no clear difference |
| probe_224 | macro_f1 | 3.6 | [2.0, 5.3] | [1.6, 6.2] | 4/5 | A beats B | A beats B |
| probe_224 | balanced_accuracy | 3.5 | [1.8, 5.2] | [1.4, 6.0] | 4/5 | A beats B | A beats B |
| probe_512 | macro_f1 | 1.2 | [-0.4, 2.8] | [-0.7, 4.1] | 5/5 | no clear difference | no clear difference |
| probe_512 | balanced_accuracy | 1.3 | [-0.3, 2.8] | [-0.7, 4.2] | 4/5 | no clear difference | no clear difference |
| mean | macro_f1 | 8.3 | [6.8, 9.8] | [5.7, 12.8] | 5/5 | A beats B | A beats B |
| mean | balanced_accuracy | 8.6 | [7.1, 10.2] | [6.0, 13.5] | 5/5 | A beats B | A beats B |
| max | macro_f1 | 1.3 | [0.0, 2.5] | [-0.3, 3.5] | 3/5 | no clear difference | no clear difference |
| max | balanced_accuracy | 1.5 | [0.2, 2.7] | [-0.1, 4.0] | 4/5 | A beats B | no clear difference |

The verdict differs between the two ranges for: max (balanced_accuracy): photo range 'A beats B', block range 'no clear difference'.

## 7. Stability seeds (validation macro-F1)

Seed 0 is the declared run; seeds 1 and 2 are the declared stability seeds. Validation only.

| aggregator | seed | round | val macro-F1 |
|---|---|---|---|
| mean | 0 | 1 | 80.2 |
| mean | 0 | 2 | 80.6 |
| mean | 0 | 3 | 77.1 |
| mean | 0 | 4 | 78.8 |
| mean | 0 | 5 | 75.7 |
| mean | 0 | mean | 78.5 |
| mean | 1 | 1 | 78.7 |
| mean | 1 | 2 | 77.6 |
| mean | 1 | 3 | 78.0 |
| mean | 1 | 4 | 79.4 |
| mean | 1 | 5 | 76.7 |
| mean | 1 | mean | 78.1 |
| mean | 2 | 1 | 78.8 |
| mean | 2 | 2 | 79.5 |
| mean | 2 | 3 | 77.3 |
| mean | 2 | 4 | 79.8 |
| mean | 2 | 5 | 74.9 |
| mean | 2 | mean | 78.1 |
| mean | all | mean ± SD over seeds | 78.2 ± 0.2 |
| max | 0 | 1 | 88.8 |
| max | 0 | 2 | 83.3 |
| max | 0 | 3 | 81.5 |
| max | 0 | 4 | 86.7 |
| max | 0 | 5 | 83.5 |
| max | 0 | mean | 84.8 |
| max | 1 | 1 | 85.9 |
| max | 1 | 2 | 82.8 |
| max | 1 | 3 | 81.0 |
| max | 1 | 4 | 86.0 |
| max | 1 | 5 | 83.5 |
| max | 1 | mean | 83.8 |
| max | 2 | 1 | 87.3 |
| max | 2 | 2 | 82.9 |
| max | 2 | 3 | 81.6 |
| max | 2 | 4 | 87.0 |
| max | 2 | 5 | 83.2 |
| max | 2 | mean | 84.4 |
| max | all | mean ± SD over seeds | 84.3 ± 0.5 |
| attention | 0 | 1 | 86.3 |
| attention | 0 | 2 | 86.2 |
| attention | 0 | 3 | 81.8 |
| attention | 0 | 4 | 85.6 |
| attention | 0 | 5 | 83.9 |
| attention | 0 | mean | 84.7 |
| attention | 1 | 1 | 86.7 |
| attention | 1 | 2 | 82.2 |
| attention | 1 | 3 | 80.3 |
| attention | 1 | 4 | 84.8 |
| attention | 1 | 5 | 81.5 |
| attention | 1 | mean | 83.1 |
| attention | 2 | 1 | 87.4 |
| attention | 2 | 2 | 84.0 |
| attention | 2 | 3 | 80.3 |
| attention | 2 | 4 | 85.6 |
| attention | 2 | 5 | 83.1 |
| attention | 2 | mean | 84.1 |
| attention | all | mean ± SD over seeds | 84.0 ± 0.8 |
