# Partition-order experiment

Built by `src/partition_report.py` from `runs/`; do not edit by hand. Declared on 2026-11-20 (handbook §24).

**What this is.** A general lesson about testing practice. Testing mistakes are common across plant-disease recognition work (Xu et al., 2024). Here we repeat three of them on our own photos and models and measure how much each raises the score, compared with the correct protocol. No published study is assessed.

**Conditions.** A is the correct protocol: the session-blocked split, augmentation on training photos only (the five headline runs of each model, reused as they are). B makes eight copies of every photo with the same augmentation, then splits the copies at random, so copies of one photo land on both sides. C splits photos at random and ignores near-duplicate groups and capture blocks. D trains on every photo and scores on every photo. L is a reference row: the locked split, which groups pixel-level copies but not capture sessions, with the settings selected on it (not A's).

**How to read it.** B, C and D are deliberately wrong. Their scores are not estimates of how the models perform on new photos, and they are used nowhere else. All of them reuse A's selected settings: nothing is tuned. B takes as many training steps per epoch as A. B is scored on its test copies, which include augmented copies; `B_clean` keeps only the clean copy of each photo. D stops at A's best epoch and its five repeats use five seeds on the same photos. Inflation is the pooled score of a condition minus the pooled score of A. Its range is a bootstrap over source photos (resampled within class; all copies and repeats of a photo stay together). A condition is called higher (or lower) than A only when the range excludes 0 and at least 4 of 5 rounds agree. "Rounds above A" pairs round r of a condition with round r of A; for D (every repeat scores all photos) and for L (another split) this is a convention, not a comparison on identical photos. The same-camera-within-10-s share counts a photo's own training copy, so it is about 100% by construction for B and D.
One training seed per round and condition; the rounds are the repeats.

## EfficientNetV2-S, whole photo at 224 px

Macro-F1 (percent, test photos of each condition):

| condition | what it does | scored rows | macro-F1, pooled [95% range] | mean ± SD over rounds | inflation vs A (points) [95% range] | rounds above A | reading |
|---|---|---|---|---|---|---|---|
| A | correct: grouped by capture block, augmentation on training photos only | 3,334 | 75.9 [74.4, 77.3] | 74.1 ± 4.8 | – | – | reference |
| L | reference: locked split (duplicates grouped, capture sessions not), its own selected settings | 3,334 | 83.6 [82.3, 84.9] | 83.5 ± 1.8 | +7.7 [+6.1, +9.2] | 5/5 | clearly above A |
| B | augment first, then split the copies at random | 26,672 | 93.0 [92.5, 93.6] | 93.0 ± 0.5 | +17.2 [+15.8, +18.6] | 5/5 | clearly above A |
| B_clean | as B, scored on the clean copy of each photo only | 3,334 | 94.8 [94.0, 95.6] | 94.8 ± 0.9 | +18.9 [+17.5, +20.4] | 5/5 | clearly above A |
| C | no grouping: photos split at random | 3,334 | 84.6 [83.4, 85.8] | 84.6 ± 2.8 | +8.8 [+7.2, +10.3] | 5/5 | clearly above A |
| D | train on every photo, score on every photo | 16,670 | 86.9 [86.1, 87.7] | 87.0 ± 2.3 | +11.0 [+9.7, +12.3] | 5/5 | clearly above A |

Balanced accuracy:

| condition | balanced accuracy, pooled | inflation vs A (points) [95% range] | reading |
|---|---|---|---|
| A | 75.9 | – | reference |
| L | 83.6 | +7.7 [+6.1, +9.2] | clearly above A |
| B | 92.9 | +17.0 [+15.5, +18.4] | clearly above A |
| B_clean | 94.8 | +18.8 [+17.4, +20.4] | clearly above A |
| C | 84.5 | +8.6 [+7.0, +10.2] | clearly above A |
| D | 88.0 | +12.1 [+10.8, +13.5] | clearly above A |

Leak counter: share of the scored rows (all five rounds) whose photo, group, block or neighbour in time is also in training (percent):

| condition | same source photo | same near-duplicate group | same capture block | same camera within 10 s |
|---|---|---|---|---|
| A | 0.0 | 0.0 | 0.0 | 0.0 |
| L | 0.0 | 0.0 | 99.7 | 90.2 |
| B | 100.0 | 100.0 | 100.0 | 100.0 |
| B_clean | 100.0 | 100.0 | 100.0 | 100.0 |
| C | 0.0 | 11.4 | 99.8 | 91.9 |
| D | 100.0 | 100.0 | 100.0 | 100.0 |

## Main tile model (E1 tiles + gated-attention head)

Macro-F1 (percent, test photos of each condition):

| condition | what it does | scored rows | macro-F1, pooled [95% range] | mean ± SD over rounds | inflation vs A (points) [95% range] | rounds above A | reading |
|---|---|---|---|---|---|---|---|
| A | correct: grouped by capture block, augmentation on training photos only | 3,334 | 78.8 [77.3, 80.2] | 77.9 ± 3.1 | – | – | reference |
| L | reference: locked split (duplicates grouped, capture sessions not), its own selected settings | 3,334 | 82.9 [81.5, 84.2] | 82.8 ± 1.5 | +4.1 [+2.8, +5.4] | 5/5 | clearly above A |
| B | augment first, then split the copies at random | 26,672 | 90.2 [89.5, 90.9] | 90.2 ± 0.5 | +11.5 [+10.3, +12.7] | 5/5 | clearly above A |
| B_clean | as B, scored on the clean copy of each photo only | 3,334 | 91.1 [90.1, 92.1] | 91.1 ± 1.5 | +12.3 [+11.0, +13.7] | 5/5 | clearly above A |
| C | no grouping: photos split at random | 3,334 | 82.7 [81.4, 84.0] | 82.7 ± 2.0 | +4.0 [+2.8, +5.2] | 4/5 | clearly above A |
| D | train on every photo, score on every photo | 16,670 | 88.9 [88.0, 89.7] | 88.9 ± 2.6 | +10.1 [+8.9, +11.3] | 5/5 | clearly above A |

Balanced accuracy:

| condition | balanced accuracy, pooled | inflation vs A (points) [95% range] | reading |
|---|---|---|---|
| A | 79.2 | – | reference |
| L | 83.4 | +4.2 [+2.9, +5.5] | clearly above A |
| B | 90.8 | +11.6 [+10.4, +12.9] | clearly above A |
| B_clean | 91.7 | +12.5 [+11.1, +13.9] | clearly above A |
| C | 83.4 | +4.2 [+3.0, +5.4] | clearly above A |
| D | 90.4 | +11.2 [+9.9, +12.4] | clearly above A |

Leak counter: share of the scored rows (all five rounds) whose photo, group, block or neighbour in time is also in training (percent):

| condition | same source photo | same near-duplicate group | same capture block | same camera within 10 s |
|---|---|---|---|---|
| A | 0.0 | 0.0 | 0.0 | 0.0 |
| L | 0.0 | 0.0 | 99.7 | 90.2 |
| B | 100.0 | 100.0 | 100.0 | 100.0 |
| B_clean | 100.0 | 100.0 | 100.0 | 100.0 |
| C | 0.0 | 11.4 | 99.8 | 91.9 |
| D | 100.0 | 100.0 | 100.0 | 100.0 |
