# Committee against ConvNeXt-T at 448 px alone (session-blocked split, test)

Built by `src/paired_committee.py` from `runs/`; do not edit by hand. Declared on 2026-11-21. All
scores are percentages, and differences are in percentage points.

**Which photos:** test photos of the session-blocked split (`splits/session_blocked/folds.csv`). Nothing was trained
or tested again: the predictions are those of the committee `row2_plain` (tested once, `runs/committee_plan.json`) and of
the five ConvNeXt-T 448 px runs (tested once, `runs/session_conditions_plan.json`). Every test round holds the same photos in both.

**Limits to read first**

- The comparison was declared **after** both test means were seen (they are in section 1), so it is post-hoc. It
  decides no selection. Stage 11 tested the committee against attention only.
- ConvNeXt-T at 448 px is **one of the two members** of the committee, so the two are not independent. The comparison
  asks whether adding the tile model to the best whole-image model helps.
- There are 2 metrics and 2 ranges, so read a single row with that number of comparisons in mind.
- Test scores use one training seed. The two ranges are around the pooled score (every test photo once).
  "photo range" is the declared 95% photo bootstrap. "block range" resamples whole capture blocks and is wider when
  neighbouring photos are alike.

**Rule** (the decision rule with the Stage 11 refinement, stated under both ranges). "Committee beats ConvNeXt-T" needs the
95% range of the difference above 0 **and** the committee winning at least 4 of 5 rounds. "ConvNeXt-T beats committee"
needs the range below 0 and ConvNeXt-T winning at least 4 of 5 rounds (a tied round is won by neither). Otherwise: no
clear difference.

## 1. Test scores (mean ± SD over the 5 rounds, then the photo range and the block range of the pooled test predictions)

| method | macro-F1 (mean ± SD) | macro-F1 photo range | macro-F1 block range | balanced acc. (mean ± SD) | balanced acc. photo range | balanced acc. block range | ECE |
|---|---|---|---|---|---|---|---|
| Committee (ConvNeXt-T 448 px + attention) | 81.6 ± 3.6 | [81.0, 83.7] | [77.8, 84.9] | 81.4 ± 4.5 | [81.2, 83.9] | [78.2, 85.1] | 0.102 |
| ConvNeXt-T 448 px alone | 79.2 ± 3.8 | [78.8, 81.6] | [75.4, 82.6] | 79.1 ± 5.0 | [78.9, 81.8] | [75.5, 82.8] | 0.050 |

## 2. Committee against ConvNeXt-T at 448 px

| metric | difference, committee minus ConvNeXt-T (points) | photo range | block range | rounds won (committee / ConvNeXt-T) | verdict (photo range) | verdict (block range) |
|---|---|---|---|---|---|---|
| macro_f1 | +2.2 | [+1.2, +3.2] | [+1.0, +3.8] | 4/1 of 5 | committee beats ConvNeXt-T | committee beats ConvNeXt-T |
| balanced_accuracy | +2.3 | [+1.3, +3.3] | [+0.9, +4.1] | 4/1 of 5 | committee beats ConvNeXt-T | committee beats ConvNeXt-T |

Verdicts:

- macro_f1, photo range: committee beats ConvNeXt-T.
- macro_f1, block range: committee beats ConvNeXt-T.
- balanced_accuracy, photo range: committee beats ConvNeXt-T.
- balanced_accuracy, block range: committee beats ConvNeXt-T.

Table: `results/tables/paired_committee.csv`.
