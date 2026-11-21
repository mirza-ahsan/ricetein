# Session-blocked split: every method tuned on its own validation photos

Built by `src/session_report.py` from `runs/`, `splits/session_blocked/folds.csv` and the follow-up tables;
do not edit by hand. Protocol declared 2026-11-16. All scores are percentages.

**Which photos each number comes from**

- Sections 1 and 6 are **validation** scores of the session-blocked split (used for selection only).
- Sections 2, 3 and 5 are **test** scores of the session-blocked split (`splits/session_blocked/folds.csv`).
- Section 4 shows test scores of both splits side by side.
- "mean ± SD" is over the five rounds. Both ranges are around the pooled score (every test photo once), so a
  mean can sit slightly outside its range when the rounds differ a lot.
- "photo range" is the declared 95% photo bootstrap. "block range" resamples whole capture blocks (photos
  shot without a pause of more than 60 s from one camera) and is wider when neighbouring photos are alike.

**Limits to read first**

- The session-blocked split was declared on 2026-11-14, **after** the test results of the locked split were seen.
- Before this protocol was declared (2026-11-16), the session-split test photos had been scored once, with the
  settings chosen on the locked split (`docs/reports/followup_report.md`, section 2). Every selection here
  uses session-split validation scores only, and each selected configuration was tested once.
- Class shares on the session split are reported, not enforced (`docs/reports/session_split_report.md`).
- Tuning budgets differ between method families: each of the five head methods keeps the better of two
  40-trial studies, a CNN has three learning rates plus up to 3 added ones, and a probe has six
  L2 strengths plus up to 3 added ones. Within a family the budget is the same.
  The head studies are widened only towards higher learning rates; a CNN or probe grid can grow at either end.
- A photo that is a validation photo in one round is a test photo in another round (41.9% of the photos). Settings
  chosen on the five-round validation mean are therefore not independent of the pooled test photos. This holds
  for every method, and a larger search can gain more from it.
- Test scores use one training seed (seed 0). The stability seeds are validation only and cover mean, max and
  attention, so the "4 of 5 rounds" rule does not include seed-to-seed variation.
- Section 1 column "selected runs differ from those tested on 2026-11-16": "no" means the search chose the
  locked-split setting again, so that method's test score was already known before this protocol was declared.
  It is always "yes" for the five head methods, whose runs are new.
- The photo range is stratified by class; the block range resamples whole blocks and is not, so it also
  contains class-count variation. That is part of why it is wider.

## 1. Searches (validation macro-F1, mean over 5 rounds)

| method | value tried | mean val macro-F1 | in declared grid / study | selected | selected runs differ from those tested on 2026-11-16 |
|---|---|---|---|---|---|
| inceptionv3_224 | 3e-05 | 65.1 | yes |  | yes |
| inceptionv3_224 | 0.0001 | 68.2 | yes |  | yes |
| inceptionv3_224 | 0.0003 | 72.8 | yes |  | yes |
| inceptionv3_224 | 0.001 | 73.1 | added | yes | yes |
| inceptionv3_224 | 0.003 | 70.7 | added |  | yes |
| efficientnetv2s_224 | 3e-05 | 64.9 | yes |  | yes |
| efficientnetv2s_224 | 0.0001 | 72.1 | yes |  | yes |
| efficientnetv2s_224 | 0.0003 | 74.4 | yes |  | yes |
| efficientnetv2s_224 | 0.001 | 75.9 | added | yes | yes |
| efficientnetv2s_224 | 0.003 | 75.1 | added |  | yes |
| convnext_tiny_224 | 3e-05 | 76.1 | yes |  | no |
| convnext_tiny_224 | 0.0001 | 76.1 | yes |  | no |
| convnext_tiny_224 | 0.0003 | 77.2 | yes | yes | no |
| convnext_tiny_224 | 0.001 | 76.5 | added |  | no |
| swin_tiny_224 | 3e-05 | 76.0 | yes |  | yes |
| swin_tiny_224 | 0.0001 | 77.1 | yes | yes | yes |
| swin_tiny_224 | 0.0003 | 75.3 | yes |  | yes |
| probe_224 | 0.001 | 65.6 | yes |  | yes |
| probe_224 | 0.01 | 65.8 | yes |  | yes |
| probe_224 | 0.1 | 66.5 | yes |  | yes |
| probe_224 | 1 | 67.6 | yes |  | yes |
| probe_224 | 10 | 68.5 | yes |  | yes |
| probe_224 | 100 | 69.9 | yes |  | yes |
| probe_224 | 1000 | 69.9 | added | yes | yes |
| probe_224 | 10000 | 67.9 | added |  | yes |
| probe_512 | 0.001 | 65.9 | yes |  | no |
| probe_512 | 0.01 | 66.1 | yes |  | no |
| probe_512 | 0.1 | 66.4 | yes |  | no |
| probe_512 | 1 | 67.8 | yes |  | no |
| probe_512 | 10 | 70.4 | yes |  | no |
| probe_512 | 100 | 71.8 | yes | yes | no |
| probe_512 | 1000 | 70.2 | added |  | no |
| mean | declared | 69.7 | yes |  | yes |
| mean | wide | 72.1 | added | yes | yes |
| max | declared | 76.3 | yes |  | yes |
| max | wide | 76.6 | added | yes | yes |
| attention | declared | 78.5 | yes | yes | yes |
| attention | wide | 77.7 | added |  | yes |
| whole224 | declared | 72.4 | yes |  | yes |
| whole224 | wide | 73.2 | added | yes | yes |
| whole512 | declared | 75.3 | yes | yes | yes |
| whole512 | wide | 75.2 | added |  | yes |

## 2. Results on the session-blocked split (test)

| method | macro-F1 (mean ± SD) | macro-F1 photo range | macro-F1 block range | balanced acc. (mean ± SD) | balanced acc. photo range | balanced acc. block range | ECE |
|---|---|---|---|---|---|---|---|
| inceptionv3_224 | 72.0 ± 3.6 | [71.2, 74.3] | [68.1, 75.6] | 72.2 ± 4.9 | [71.9, 75.1] | [69.0, 76.4] | 0.072 |
| efficientnetv2s_224 | 74.1 ± 4.8 | [74.3, 77.3] | [69.8, 78.8] | 73.9 ± 5.9 | [74.3, 77.4] | [69.5, 79.0] | 0.035 |
| convnext_tiny_224 | 75.0 ± 5.0 | [74.4, 77.4] | [70.5, 79.1] | 75.8 ± 3.7 | [75.2, 78.3] | [71.9, 79.6] | 0.067 |
| swin_tiny_224 | 76.3 ± 4.1 | [75.8, 78.8] | [72.3, 79.8] | 75.8 ± 4.8 | [75.6, 78.7] | [71.4, 79.6] | 0.060 |
| probe_224 | 73.0 ± 4.0 | [72.6, 75.6] | [68.9, 76.5] | 73.3 ± 4.0 | [73.3, 76.4] | [69.3, 77.1] | 0.031 |
| probe_512 | 73.5 ± 5.9 | [74.0, 77.1] | [69.2, 78.4] | 73.9 ± 6.8 | [74.4, 77.7] | [69.5, 78.9] | 0.058 |
| mean | 69.1 ± 8.5 | [69.8, 73.1] | [63.0, 75.6] | 68.9 ± 9.2 | [69.6, 73.1] | [62.5, 75.6] | 0.088 |
| max | 73.4 ± 7.1 | [74.1, 77.2] | [69.1, 78.9] | 74.0 ± 8.4 | [74.4, 77.6] | [69.0, 79.4] | 0.075 |
| attention | 77.9 ± 3.1 | [77.3, 80.2] | [74.7, 80.9] | 78.0 ± 3.6 | [77.6, 80.6] | [75.3, 81.5] | 0.051 |
| whole224 | 69.3 ± 7.1 | [69.1, 72.3] | [65.0, 73.6] | 69.1 ± 7.2 | [69.2, 72.5] | [65.1, 73.7] | 0.125 |
| whole512 | 73.2 ± 6.9 | [73.2, 76.4] | [68.7, 78.0] | 73.5 ± 6.9 | [73.8, 77.0] | [69.1, 78.5] | 0.050 |

## 3. Attention against every other method (session-blocked split, test)

| A (attention) vs | metric | difference (points) | photo range | block range | rounds won | verdict (photo range) | verdict (block range) |
|---|---|---|---|---|---|---|---|
| inceptionv3_224 | macro_f1 | 5.9 | [4.2, 7.5] | [3.3, 8.8] | 5/5 | A beats B | A beats B |
| inceptionv3_224 | balanced_accuracy | 5.7 | [3.8, 7.3] | [2.8, 8.9] | 5/5 | A beats B | A beats B |
| efficientnetv2s_224 | macro_f1 | 2.9 | [1.2, 4.5] | [0.6, 6.2] | 5/5 | A beats B | A beats B |
| efficientnetv2s_224 | balanced_accuracy | 3.3 | [1.6, 4.9] | [0.6, 7.4] | 5/5 | A beats B | A beats B |
| convnext_tiny_224 | macro_f1 | 2.8 | [1.2, 4.3] | [-0.1, 6.1] | 5/5 | A beats B | no clear difference |
| convnext_tiny_224 | balanced_accuracy | 2.3 | [0.8, 3.9] | [-0.6, 5.5] | 4/5 | A beats B | no clear difference |
| swin_tiny_224 | macro_f1 | 1.4 | [-0.1, 2.9] | [-0.1, 3.5] | 5/5 | no clear difference | no clear difference |
| swin_tiny_224 | balanced_accuracy | 2.0 | [0.5, 3.6] | [0.3, 5.0] | 5/5 | A beats B | A beats B |
| probe_224 | macro_f1 | 4.6 | [3.0, 6.3] | [2.6, 7.4] | 5/5 | A beats B | A beats B |
| probe_224 | balanced_accuracy | 4.3 | [2.6, 6.0] | [2.1, 7.6] | 5/5 | A beats B | A beats B |
| probe_512 | macro_f1 | 3.1 | [1.6, 4.8] | [0.9, 6.7] | 5/5 | A beats B | A beats B |
| probe_512 | balanced_accuracy | 3.1 | [1.6, 4.7] | [0.6, 7.4] | 4/5 | A beats B | A beats B |
| mean | macro_f1 | 7.2 | [5.7, 8.8] | [4.1, 12.8] | 5/5 | A beats B | A beats B |
| mean | balanced_accuracy | 7.8 | [6.2, 9.4] | [4.0, 14.2] | 5/5 | A beats B | A beats B |
| max | macro_f1 | 3.0 | [1.7, 4.4] | [0.8, 6.5] | 5/5 | A beats B | A beats B |
| max | balanced_accuracy | 3.1 | [1.8, 4.6] | [0.1, 7.9] | 3/5 | no clear difference | no clear difference |
| whole224 | macro_f1 | 8.0 | [6.2, 9.7] | [5.5, 11.2] | 5/5 | A beats B | A beats B |
| whole224 | balanced_accuracy | 8.4 | [6.6, 10.0] | [5.6, 12.1] | 5/5 | A beats B | A beats B |
| whole512 | macro_f1 | 3.9 | [2.3, 5.6] | [1.5, 7.5] | 5/5 | A beats B | A beats B |
| whole512 | balanced_accuracy | 3.8 | [2.2, 5.5] | [1.1, 7.6] | 5/5 | A beats B | A beats B |

The verdict differs between the two ranges for: convnext_tiny_224 (macro_f1): photo range 'A beats B', block range 'no clear difference'; convnext_tiny_224 (balanced_accuracy): photo range 'A beats B', block range 'no clear difference'.

## 4. Three set-ups side by side (test macro-F1, mean ± SD over 5 rounds)

The locked split column is the final locked-split configuration of the follow-up study. The middle column
refits the nine core settings chosen on the locked split; the whole-photo controls were not part of it.

| method | locked split | session split, locked-split settings | session split, tuned on it |
|---|---|---|---|
| inceptionv3_224 | 80.9 ± 2.7 | 72.7 ± 4.3 | 72.0 ± 3.6 |
| efficientnetv2s_224 | 83.5 ± 1.8 | 71.7 ± 7.4 | 74.1 ± 4.8 |
| convnext_tiny_224 | 83.4 ± 2.6 | 75.0 ± 5.0 | 75.0 ± 5.0 |
| swin_tiny_224 | 83.7 ± 1.3 | 75.9 ± 3.3 | 76.3 ± 4.1 |
| probe_224 | 78.2 ± 0.7 | 72.0 ± 4.6 | 73.0 ± 4.0 |
| probe_512 | 81.3 ± 1.1 | 73.5 ± 5.9 | 73.5 ± 5.9 |
| mean | 78.5 ± 2.7 | 66.2 ± 8.7 | 69.1 ± 8.5 |
| max | 82.9 ± 1.7 | 73.9 ± 6.7 | 73.4 ± 7.1 |
| attention | 82.8 ± 1.5 | 75.7 ± 4.6 | 77.9 ± 3.1 |
| whole224 | 80.2 ± 1.8 | — | 69.3 ± 7.1 |
| whole512 | 81.8 ± 1.8 | — | 73.2 ± 6.9 |

## 5. Shortcut controls on the session-blocked split (test, mean ± SD over 5 rounds)

| baseline | macro-F1 | balanced acc. |
|---|---|---|
| majority | 15.4 ± 1.1 | 25.0 ± 0.0 |
| random_prior | 25.2 ± 1.6 | 25.6 ± 1.7 |
| colour_only | 43.5 ± 2.8 | 45.2 ± 3.6 |
| background_only | 39.7 ± 3.6 | 41.8 ± 4.3 |
| properties_only | 29.5 ± 3.1 | 33.1 ± 3.5 |
| camera_only | 15.1 ± 5.1 | 30.5 ± 4.2 |
| time_neighbour | 30.4 ± 10.8 | 33.7 ± 12.1 |
| background_strict | 38.9 ± 4.9 | 43.3 ± 5.4 |

## 6. Stability seeds of the selected tile studies (validation macro-F1)

Seed 0 is the declared run; seeds 1 and 2 are the declared stability seeds. Validation only.

| aggregator | seed | round | val macro-F1 |
|---|---|---|---|
| mean | 0 | 1 | 62.4 |
| mean | 0 | 2 | 78.5 |
| mean | 0 | 3 | 72.8 |
| mean | 0 | 4 | 65.4 |
| mean | 0 | 5 | 81.7 |
| mean | 0 | mean | 72.1 |
| mean | 1 | 1 | 61.3 |
| mean | 1 | 2 | 76.8 |
| mean | 1 | 3 | 71.7 |
| mean | 1 | 4 | 66.2 |
| mean | 1 | 5 | 84.2 |
| mean | 1 | mean | 72.1 |
| mean | 2 | 1 | 56.0 |
| mean | 2 | 2 | 76.0 |
| mean | 2 | 3 | 69.2 |
| mean | 2 | 4 | 64.2 |
| mean | 2 | 5 | 83.0 |
| mean | 2 | mean | 69.7 |
| mean | all | mean ± SD over seeds | 71.3 ± 1.4 |
| max | 0 | 1 | 66.8 |
| max | 0 | 2 | 80.4 |
| max | 0 | 3 | 76.5 |
| max | 0 | 4 | 73.9 |
| max | 0 | 5 | 85.3 |
| max | 0 | mean | 76.6 |
| max | 1 | 1 | 65.8 |
| max | 1 | 2 | 78.2 |
| max | 1 | 3 | 71.9 |
| max | 1 | 4 | 72.2 |
| max | 1 | 5 | 84.7 |
| max | 1 | mean | 74.6 |
| max | 2 | 1 | 69.1 |
| max | 2 | 2 | 79.6 |
| max | 2 | 3 | 71.3 |
| max | 2 | 4 | 74.3 |
| max | 2 | 5 | 83.2 |
| max | 2 | mean | 75.5 |
| max | all | mean ± SD over seeds | 75.5 ± 1.0 |
| attention | 0 | 1 | 72.8 |
| attention | 0 | 2 | 83.0 |
| attention | 0 | 3 | 72.6 |
| attention | 0 | 4 | 78.9 |
| attention | 0 | 5 | 85.2 |
| attention | 0 | mean | 78.5 |
| attention | 1 | 1 | 71.9 |
| attention | 1 | 2 | 81.8 |
| attention | 1 | 3 | 73.5 |
| attention | 1 | 4 | 74.4 |
| attention | 1 | 5 | 85.1 |
| attention | 1 | mean | 77.4 |
| attention | 2 | 1 | 70.5 |
| attention | 2 | 2 | 81.0 |
| attention | 2 | 3 | 72.1 |
| attention | 2 | 4 | 75.8 |
| attention | 2 | 5 | 85.0 |
| attention | 2 | mean | 76.9 |
| attention | all | mean ± SD over seeds | 77.6 ± 0.8 |
