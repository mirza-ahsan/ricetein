# Attention against the nine Stage 5 conditions (session-blocked split, test)

Built by `src/paired_conditions.py` from `runs/`; do not edit by hand. Declared on 2026-11-21 (handbook §24). All
scores are percentages, and differences are in percentage points.

**Which photos:** test photos of the session-blocked split (`splits/session_blocked/folds.csv`). Nothing was trained
or tested again: the predictions are those of the attention runs (tested 2026-11-17, `runs/session_evaluation_plan.json`)
and of the nine conditions (tested 2026-11-21, `runs/session_conditions_plan.json`).

**Limits to read first**

- The comparison was declared **after** both methods' test means were seen (ConvNeXt-T at 448 px has the higher mean),
  so it is post-hoc. It decides no selection.
- There are 9 conditions, 2 metrics and 2 ranges, so read a single row with that number of comparisons in mind.
- Test scores use one training seed. The two ranges are around the pooled score (every test photo once).
  "photo range" is the declared 95% photo bootstrap. "block range" resamples whole capture blocks and is wider when
  neighbouring photos are alike.
- The conditions use the learning rate selected at 224 px; they were not tuned at their own size. Attention was tuned
  with its own search (`docs/reports/session_report.md`).

**Rule** (the decision rule, stated under both ranges). "Attention beats it" needs the 95% range of attention minus the
condition above 0 **and** attention winning at least 4 of 5 rounds. "It beats attention" is the same test with the sides
swapped. Otherwise: no clear difference. A claim that attention (or a condition) is best is made only where this rule
supports it.

## 1. Test scores (mean ± SD over the 5 rounds, then the photo range and the block range of the pooled test predictions)

| method | macro-F1 (mean ± SD) | macro-F1 photo range | macro-F1 block range | balanced acc. (mean ± SD) | balanced acc. photo range | balanced acc. block range | ECE |
|---|---|---|---|---|---|---|---|
| attention | 77.9 ± 3.1 | [77.3, 80.2] | [74.7, 80.9] | 78.0 ± 3.6 | [77.6, 80.6] | [75.3, 81.5] | 0.051 |
| ConvNeXt-T, 448 px | 79.2 ± 3.8 | [78.8, 81.6] | [75.4, 82.6] | 79.1 ± 5.0 | [78.9, 81.8] | [75.5, 82.8] | 0.050 |
| InceptionV3, 128 px | 56.9 ± 6.4 | [57.1, 60.5] | [52.5, 62.5] | 56.7 ± 5.5 | [56.6, 60.1] | [52.1, 62.0] | 0.056 |
| InceptionV3, 224 px + Otsu | 71.3 ± 3.4 | [70.9, 74.1] | [66.8, 75.4] | 70.8 ± 3.9 | [70.6, 74.0] | [66.4, 75.4] | 0.060 |
| EfficientNetV2-S, 128 px | 62.8 ± 4.7 | [63.2, 66.6] | [57.3, 68.5] | 62.4 ± 6.1 | [63.1, 66.7] | [57.1, 68.7] | 0.053 |
| EfficientNetV2-S, 224 px + Otsu | 70.2 ± 2.5 | [70.0, 73.0] | [65.0, 75.0] | 70.6 ± 4.2 | [70.8, 74.0] | [66.3, 75.6] | 0.055 |
| ConvNeXt-T, 128 px | 67.8 ± 6.0 | [67.8, 71.0] | [63.5, 72.4] | 67.8 ± 6.4 | [67.6, 71.0] | [63.8, 72.4] | 0.082 |
| ConvNeXt-T, 224 px + Otsu | 73.5 ± 3.4 | [72.9, 75.8] | [69.7, 76.9] | 74.2 ± 3.3 | [74.0, 77.0] | [70.7, 77.9] | 0.071 |
| Swin-T, 128 px | 65.2 ± 4.4 | [64.9, 68.2] | [61.7, 69.0] | 65.6 ± 4.9 | [65.2, 68.6] | [61.6, 69.7] | 0.081 |
| Swin-T, 224 px + Otsu | 71.7 ± 3.8 | [71.5, 74.5] | [66.9, 75.9] | 72.3 ± 4.9 | [72.0, 75.2] | [67.3, 76.6] | 0.031 |

## 2. Attention against each condition

| condition | metric | difference, attention minus condition (points) | photo range | block range | rounds won (attention / condition) | verdict (photo range) | verdict (block range) |
|---|---|---|---|---|---|---|---|
| ConvNeXt-T, 448 px | macro_f1 | -1.5 | [-3.0, 0.2] | [-3.3, 0.9] | 1/4 of 5 | no clear difference | no clear difference |
| ConvNeXt-T, 448 px | balanced_accuracy | -1.2 | [-2.8, 0.5] | [-3.2, 1.8] | 1/4 of 5 | no clear difference | no clear difference |
| InceptionV3, 128 px | macro_f1 | 19.9 | [18.1, 21.9] | [16.4, 24.3] | 5/0 of 5 | attention beats it | attention beats it |
| InceptionV3, 128 px | balanced_accuracy | 20.8 | [18.9, 22.8] | [16.9, 25.6] | 5/0 of 5 | attention beats it | attention beats it |
| InceptionV3, 224 px + Otsu | macro_f1 | 6.2 | [4.5, 7.9] | [3.5, 10.0] | 5/0 of 5 | attention beats it | attention beats it |
| InceptionV3, 224 px + Otsu | balanced_accuracy | 6.9 | [5.1, 8.6] | [3.7, 11.2] | 5/0 of 5 | attention beats it | attention beats it |
| EfficientNetV2-S, 128 px | macro_f1 | 13.8 | [12.0, 15.7] | [10.7, 19.0] | 5/0 of 5 | attention beats it | attention beats it |
| EfficientNetV2-S, 128 px | balanced_accuracy | 14.3 | [12.4, 16.1] | [10.7, 20.2] | 5/0 of 5 | attention beats it | attention beats it |
| EfficientNetV2-S, 224 px + Otsu | macro_f1 | 7.2 | [5.5, 8.9] | [4.7, 11.0] | 5/0 of 5 | attention beats it | attention beats it |
| EfficientNetV2-S, 224 px + Otsu | balanced_accuracy | 6.8 | [5.1, 8.5] | [4.3, 10.4] | 5/0 of 5 | attention beats it | attention beats it |
| ConvNeXt-T, 128 px | macro_f1 | 9.3 | [7.6, 11.0] | [6.7, 12.7] | 5/0 of 5 | attention beats it | attention beats it |
| ConvNeXt-T, 128 px | balanced_accuracy | 9.9 | [8.1, 11.6] | [7.0, 13.5] | 5/0 of 5 | attention beats it | attention beats it |
| ConvNeXt-T, 224 px + Otsu | macro_f1 | 4.3 | [2.6, 6.0] | [2.1, 7.1] | 5/0 of 5 | attention beats it | attention beats it |
| ConvNeXt-T, 224 px + Otsu | balanced_accuracy | 3.7 | [1.9, 5.3] | [1.3, 6.8] | 5/0 of 5 | attention beats it | attention beats it |
| Swin-T, 128 px | macro_f1 | 12.2 | [10.4, 14.0] | [9.4, 15.7] | 5/0 of 5 | attention beats it | attention beats it |
| Swin-T, 128 px | balanced_accuracy | 12.3 | [10.5, 14.1] | [9.2, 16.3] | 5/0 of 5 | attention beats it | attention beats it |
| Swin-T, 224 px + Otsu | macro_f1 | 5.7 | [4.0, 7.2] | [3.1, 9.4] | 5/0 of 5 | attention beats it | attention beats it |
| Swin-T, 224 px + Otsu | balanced_accuracy | 5.5 | [3.8, 7.1] | [2.5, 9.8] | 5/0 of 5 | attention beats it | attention beats it |

Verdicts over the nine conditions:

- macro_f1, photo range: 8 attention beats it, 1 no clear difference, 0 it beats attention.
- macro_f1, block range: 8 attention beats it, 1 no clear difference, 0 it beats attention.
- balanced_accuracy, photo range: 8 attention beats it, 1 no clear difference, 0 it beats attention.
- balanced_accuracy, block range: 8 attention beats it, 1 no clear difference, 0 it beats attention.

Table: `results/tables/paired_conditions.csv`.
