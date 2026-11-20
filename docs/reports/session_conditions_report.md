# Stage 5 other conditions on the session-blocked split

Declared on 2026-11-20 (handbook §24) before any of these runs. Each of the four whole-image models is trained at
128 px and at 224 px with Otsu background removal; the model with the best validation score at 224 px
(convnext_tiny) is also trained at 448 px. Every model keeps the learning rate selected at 224 px on validation
photos, and the recipe and budget are those of the 224 px runs. Nothing was searched again. Every condition was
tested once, on the test photos of the 5 rounds. The 224 px rows are the runs of `docs/reports/session_report.md`.

Scores are test macro-F1 and balanced accuracy: mean ± SD over the 5 rounds, then the declared 95% photo range
and the 95% block range of the pooled test predictions (a block is a run of photos from one camera with no pause over
60 s; blocks are resampled together). In each comparison A is the run named first in the row: "A beats B" needs the
range above 0 **and** at least 4 of 5 round wins, and a row that fails either is "no clear difference". Differences are
in percentage points (A minus B). There are 9 conditions, 2 scores and 2 directions, so read a single row with that
number of comparisons in mind.

## Scores of every model and condition

| method | macro-F1 (mean ± SD) | macro-F1 photo range | macro-F1 block range | balanced acc. (mean ± SD) | balanced acc. photo range | balanced acc. block range | ECE |
|---|---|---|---|---|---|---|---|
| inceptionv3, 224 px | 72.0 ± 3.6 | [71.2, 74.3] | [68.1, 75.6] | 72.2 ± 4.9 | [71.9, 75.1] | [69.0, 76.4] | 0.072 |
| inceptionv3, 128 px | 56.9 ± 6.4 | [57.1, 60.5] | [52.5, 62.5] | 56.7 ± 5.5 | [56.6, 60.1] | [52.1, 62.0] | 0.056 |
| inceptionv3, 224 px + Otsu | 71.3 ± 3.4 | [70.9, 74.1] | [66.8, 75.4] | 70.8 ± 3.9 | [70.6, 74.0] | [66.4, 75.4] | 0.060 |
| efficientnetv2s, 224 px | 74.1 ± 4.8 | [74.3, 77.3] | [69.8, 78.8] | 73.9 ± 5.9 | [74.3, 77.4] | [69.5, 79.0] | 0.035 |
| efficientnetv2s, 128 px | 62.8 ± 4.7 | [63.2, 66.6] | [57.3, 68.5] | 62.4 ± 6.1 | [63.1, 66.7] | [57.1, 68.7] | 0.053 |
| efficientnetv2s, 224 px + Otsu | 70.2 ± 2.5 | [70.0, 73.0] | [65.0, 75.0] | 70.6 ± 4.2 | [70.8, 74.0] | [66.3, 75.6] | 0.055 |
| convnext_tiny, 224 px | 75.0 ± 5.0 | [74.4, 77.4] | [70.5, 79.1] | 75.8 ± 3.7 | [75.2, 78.3] | [71.9, 79.6] | 0.067 |
| convnext_tiny, 128 px | 67.8 ± 6.0 | [67.8, 71.0] | [63.5, 72.4] | 67.8 ± 6.4 | [67.6, 71.0] | [63.8, 72.4] | 0.082 |
| convnext_tiny, 224 px + Otsu | 73.5 ± 3.4 | [72.9, 75.8] | [69.7, 76.9] | 74.2 ± 3.3 | [74.0, 77.0] | [70.7, 77.9] | 0.071 |
| convnext_tiny, 448 px | 79.2 ± 3.8 | [78.8, 81.6] | [75.4, 82.6] | 79.1 ± 5.0 | [78.9, 81.8] | [75.5, 82.8] | 0.050 |
| swin_tiny, 224 px | 76.3 ± 4.1 | [75.8, 78.8] | [72.3, 79.8] | 75.8 ± 4.8 | [75.6, 78.7] | [71.4, 79.6] | 0.060 |
| swin_tiny, 128 px | 65.2 ± 4.4 | [64.9, 68.2] | [61.7, 69.0] | 65.6 ± 4.9 | [65.2, 68.6] | [61.6, 69.7] | 0.081 |
| swin_tiny, 224 px + Otsu | 71.7 ± 3.8 | [71.5, 74.5] | [66.9, 75.9] | 72.3 ± 4.9 | [72.0, 75.2] | [67.3, 76.6] | 0.031 |

## Does the condition beat the same model at 224 px?

| condition against 224 px of the same model | metric | difference (points) | photo range | block range | rounds won | verdict (photo range) | verdict (block range) |
|---|---|---|---|---|---|---|---|
| inceptionv3, 128 px: condition beats 224 px | macro_f1 | -14.0 | [-16.0, -12.3] | [-17.3, -11.3] | 0/5 | no clear difference | no clear difference |
| inceptionv3, 128 px: condition beats 224 px | balanced_accuracy | -15.1 | [-17.2, -13.3] | [-18.5, -12.6] | 0/5 | no clear difference | no clear difference |
| inceptionv3, 224 px + Otsu: condition beats 224 px | macro_f1 | -0.3 | [-2.0, 1.4] | [-3.8, 2.1] | 2/5 | no clear difference | no clear difference |
| inceptionv3, 224 px + Otsu: condition beats 224 px | balanced_accuracy | -1.2 | [-3.0, 0.5] | [-5.2, 1.4] | 2/5 | no clear difference | no clear difference |
| efficientnetv2s, 128 px: condition beats 224 px | macro_f1 | -10.9 | [-12.6, -9.2] | [-14.7, -8.4] | 0/5 | no clear difference | no clear difference |
| efficientnetv2s, 128 px: condition beats 224 px | balanced_accuracy | -11.0 | [-12.7, -9.1] | [-15.0, -8.4] | 0/5 | no clear difference | no clear difference |
| efficientnetv2s, 224 px + Otsu: condition beats 224 px | macro_f1 | -4.4 | [-5.9, -2.8] | [-6.6, -2.4] | 1/5 | no clear difference | no clear difference |
| efficientnetv2s, 224 px + Otsu: condition beats 224 px | balanced_accuracy | -3.6 | [-5.2, -2.0] | [-5.5, -1.4] | 1/5 | no clear difference | no clear difference |
| convnext_tiny, 128 px: condition beats 224 px | macro_f1 | -6.6 | [-8.2, -4.8] | [-9.2, -4.2] | 0/5 | no clear difference | no clear difference |
| convnext_tiny, 128 px: condition beats 224 px | balanced_accuracy | -7.5 | [-9.2, -5.8] | [-10.2, -5.0] | 0/5 | no clear difference | no clear difference |
| convnext_tiny, 224 px + Otsu: condition beats 224 px | macro_f1 | -1.6 | [-3.0, -0.1] | [-3.7, 0.8] | 1/5 | no clear difference | no clear difference |
| convnext_tiny, 224 px + Otsu: condition beats 224 px | balanced_accuracy | -1.3 | [-2.7, 0.1] | [-3.5, 0.7] | 0/5 | no clear difference | no clear difference |
| swin_tiny, 128 px: condition beats 224 px | macro_f1 | -10.8 | [-12.6, -9.1] | [-13.5, -8.2] | 0/5 | no clear difference | no clear difference |
| swin_tiny, 128 px: condition beats 224 px | balanced_accuracy | -10.3 | [-12.0, -8.5] | [-13.0, -7.4] | 0/5 | no clear difference | no clear difference |
| swin_tiny, 224 px + Otsu: condition beats 224 px | macro_f1 | -4.3 | [-5.6, -2.8] | [-6.7, -2.4] | 0/5 | no clear difference | no clear difference |
| swin_tiny, 224 px + Otsu: condition beats 224 px | balanced_accuracy | -3.5 | [-4.9, -2.0] | [-5.7, -1.6] | 0/5 | no clear difference | no clear difference |
| convnext_tiny, 448 px: condition beats 224 px | macro_f1 | 4.3 | [2.7, 5.8] | [1.5, 7.4] | 5/5 | A beats B | A beats B |
| convnext_tiny, 448 px: condition beats 224 px | balanced_accuracy | 3.5 | [2.0, 5.1] | [0.9, 6.0] | 5/5 | A beats B | A beats B |

The verdict is the same under both ranges in every row.

## Does 224 px beat the condition?

The same test with the two sides swapped (the ranges are mirrored; only the round wins can differ, through ties).

| 224 px against the condition | metric | difference (points) | photo range | block range | rounds won | verdict (photo range) | verdict (block range) |
|---|---|---|---|---|---|---|---|
| inceptionv3, 128 px: 224 px beats condition | macro_f1 | 14.0 | [12.3, 16.0] | [11.3, 17.3] | 5/5 | A beats B | A beats B |
| inceptionv3, 128 px: 224 px beats condition | balanced_accuracy | 15.1 | [13.3, 17.2] | [12.6, 18.5] | 5/5 | A beats B | A beats B |
| inceptionv3, 224 px + Otsu: 224 px beats condition | macro_f1 | 0.3 | [-1.4, 2.0] | [-2.1, 3.8] | 3/5 | no clear difference | no clear difference |
| inceptionv3, 224 px + Otsu: 224 px beats condition | balanced_accuracy | 1.2 | [-0.5, 3.0] | [-1.4, 5.2] | 3/5 | no clear difference | no clear difference |
| efficientnetv2s, 128 px: 224 px beats condition | macro_f1 | 10.9 | [9.2, 12.6] | [8.4, 14.7] | 5/5 | A beats B | A beats B |
| efficientnetv2s, 128 px: 224 px beats condition | balanced_accuracy | 11.0 | [9.1, 12.7] | [8.4, 15.0] | 5/5 | A beats B | A beats B |
| efficientnetv2s, 224 px + Otsu: 224 px beats condition | macro_f1 | 4.4 | [2.8, 5.9] | [2.4, 6.6] | 4/5 | A beats B | A beats B |
| efficientnetv2s, 224 px + Otsu: 224 px beats condition | balanced_accuracy | 3.6 | [2.0, 5.2] | [1.4, 5.5] | 4/5 | A beats B | A beats B |
| convnext_tiny, 128 px: 224 px beats condition | macro_f1 | 6.6 | [4.8, 8.2] | [4.2, 9.2] | 5/5 | A beats B | A beats B |
| convnext_tiny, 128 px: 224 px beats condition | balanced_accuracy | 7.5 | [5.8, 9.2] | [5.0, 10.2] | 5/5 | A beats B | A beats B |
| convnext_tiny, 224 px + Otsu: 224 px beats condition | macro_f1 | 1.6 | [0.1, 3.0] | [-0.8, 3.7] | 4/5 | A beats B | no clear difference |
| convnext_tiny, 224 px + Otsu: 224 px beats condition | balanced_accuracy | 1.3 | [-0.1, 2.7] | [-0.7, 3.5] | 5/5 | no clear difference | no clear difference |
| swin_tiny, 128 px: 224 px beats condition | macro_f1 | 10.8 | [9.1, 12.6] | [8.2, 13.5] | 5/5 | A beats B | A beats B |
| swin_tiny, 128 px: 224 px beats condition | balanced_accuracy | 10.3 | [8.5, 12.0] | [7.4, 13.0] | 5/5 | A beats B | A beats B |
| swin_tiny, 224 px + Otsu: 224 px beats condition | macro_f1 | 4.3 | [2.8, 5.6] | [2.4, 6.7] | 5/5 | A beats B | A beats B |
| swin_tiny, 224 px + Otsu: 224 px beats condition | balanced_accuracy | 3.5 | [2.0, 4.9] | [1.6, 5.7] | 5/5 | A beats B | A beats B |
| convnext_tiny, 448 px: 224 px beats condition | macro_f1 | -4.3 | [-5.8, -2.7] | [-7.4, -1.5] | 0/5 | no clear difference | no clear difference |
| convnext_tiny, 448 px: 224 px beats condition | balanced_accuracy | -3.5 | [-5.1, -2.0] | [-6.0, -0.9] | 0/5 | no clear difference | no clear difference |

The verdict differs between the two ranges for: convnext_tiny, 224 px + Otsu: 224 px beats condition (macro_f1): photo range 'A beats B', block range 'no clear difference'.

Tables: `results/tables/session_conditions_scores.csv` and `results/tables/session_conditions_paired.csv`.
