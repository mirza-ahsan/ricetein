# Committee: does combining models beat the best single model? (Stage 11)

Built by `src/committee.py` from `runs/`; do not edit by hand. Declared 2026-11-21 ("Stage 11
committee"), written before any committee number. All scores are percentages on the session-blocked split
(`splits/session_blocked/folds.csv`) unless the heading says Paddy Doctor.

**Which photos each number comes from.** Sections 1 and 5 are **validation** photos (used for the choice).
Sections 2, 3 and 6 are **test** photos, each committee tested once (`runs/committee_*/test_started.json`).
Section 4 is **Paddy Doctor** (8,086 photos, zero-shot, native zoom); nothing was chosen or trained on it.

**Limits to read first**

- The test scores of the already tested members were known before the choice rule was written (disclosed in the
  declaration). The rule uses validation predictions only and was fixed before any committee number was computed.
- A photo that is a validation photo in one round is a test photo in another round, so settings chosen on
  validation are not independent of the pooled test photos (the limit of the session report applies).
- Members and committees use one training seed (seed 0). "Rounds won" counts the rounds in which the committee of
  that round beats the single model of that round.
- The cost column is not measured here: a committee costs the sum of its members; Stage 13 measures it.
- Test-time augmentation was done for the best whole-image member only; it was **not** done for attention (new
  tile descriptions of every flipped and rotated test photo would be needed).

## 1. Choice on validation photos

Single members (validation macro-F1, mean over the five rounds):

| member | validation macro-F1, mean over rounds |
|---|---|
| attention_e1 | 78.5 |
| attention_e2 | 66.5 |
| attention_e3 | 76.1 |
| inceptionv3_224 | 73.1 |
| efficientnetv2s_224 | 75.9 |
| convnext_tiny_224 | 77.2 |
| swin_tiny_224 | 77.1 |
| convnext_tiny_448 | 81.1 |

Chosen committees (S = mean over the rounds of the validation macro-F1 of the average):

| committee | members (weight) | S, validation | note |
|---|---|---|---|
| 2 committee with tile models | convnext_tiny_448 + attention_e1 | 81.7 |  |
| 2 committee with tile models, weighted | convnext_tiny_448 + attention_e1 | 81.7 | same as plain, not tested again |
| 3 committee of whole-image models | convnext_tiny_448 + swin_tiny_224 | 80.1 |  |
| 3 committee of whole-image models, weighted | convnext_tiny_448 x2 + swin_tiny_224 | 82.0 |  |

## 2. Test results (session-blocked split)

| model | macro-F1 (mean ± SD) | balanced acc. (mean ± SD) |
|---|---|---|
| 1 best single model (attention, E1) | 77.9 ± 3.1 | 78.0 ± 3.6 |
| 2 committee with tile models | 81.6 ± 3.6 | 81.4 ± 4.5 |
| 3 committee of whole-image models | 80.3 ± 4.0 | 79.8 ± 5.0 |
| 3 committee of whole-image models, weighted | 80.3 ± 4.0 | 79.7 ± 5.1 |

The table file `results/tables/committee_results.csv` also holds balanced accuracy, the Paddy columns and the paired
verdicts of each row against attention.

## 3. Paired tests on test photos (A against B)

Rule (the decision rule, read in both directions): A beats B if the range is above 0 and A wins at least 4 of 5 rounds;
B beats A if the range is below 0 and B wins at least 4 (a tied round is won by neither); otherwise no clear difference.

| A | B | metric | difference (points) | photo range | block range | rounds won (A) | verdict, photo range | verdict, block range |
|---|---|---|---|---|---|---|---|---|
| 2 committee with tile models | 1 best single model (attention, E1) | macro_f1 | +3.6 | [+2.4, +4.8] | [+2.1, +5.0] | 5/5 | A beats B | A beats B |
| 2 committee with tile models | 1 best single model (attention, E1) | balanced_accuracy | +3.4 | [+2.3, +4.6] | [+1.8, +4.7] | 5/5 | A beats B | A beats B |
| 3 committee of whole-image models | 1 best single model (attention, E1) | macro_f1 | +2.5 | [+1.0, +4.0] | [+0.5, +4.1] | 5/5 | A beats B | A beats B |
| 3 committee of whole-image models | 1 best single model (attention, E1) | balanced_accuracy | +2.0 | [+0.5, +3.5] | [-0.7, +3.9] | 4/5 | A beats B | no clear difference |
| 3 committee of whole-image models, weighted | 1 best single model (attention, E1) | macro_f1 | +2.5 | [+1.0, +4.0] | [+0.4, +4.1] | 5/5 | A beats B | A beats B |
| 3 committee of whole-image models, weighted | 1 best single model (attention, E1) | balanced_accuracy | +1.9 | [+0.4, +3.4] | [-0.9, +3.8] | 4/5 | A beats B | no clear difference |
| 2 committee with tile models | 3 committee of whole-image models | macro_f1 | +1.2 | [+0.1, +2.3] | [-0.2, +2.6] | 5/5 | A beats B | no clear difference |
| 2 committee with tile models | 3 committee of whole-image models | balanced_accuracy | +1.5 | [+0.4, +2.6] | [-0.0, +3.3] | 5/5 | A beats B | no clear difference |

## 4. Paddy Doctor (native zoom, zero-shot)

Committee rows: the round committees' mean ± SD, the five-round committee, and the paired test against attention
(photo range only: Paddy photos have no capture blocks).

| model | macro-F1 (mean ± SD) | balanced acc. (mean ± SD) | five-round committee macro-F1 | difference to attention (points) | photo range | rounds won | verdict |
|---|---|---|---|---|---|---|---|
| 1 best single model (attention, E1) | 26.1 ± 4.2 | 33.8 ± 2.9 | 25.5 | – | – | – | – |
| 2 committee with tile models | 23.0 ± 5.2 | 31.3 ± 3.4 | 19.7 | -5.8 | [-6.6, -5.1] | 1/5 | attention beats committee |
| 3 committee of whole-image models | 26.4 ± 8.6 | 32.9 ± 5.1 | 22.7 | -2.8 | [-4.0, -1.7] | 2/5 | no clear difference |
| 3 committee of whole-image models, weighted | 26.8 ± 8.9 | 33.3 ± 5.5 | 22.3 | -3.3 | [-4.4, -2.1] | 2/5 | no clear difference |

## 5. Disagreement of the members on validation photos (rounds pooled)

`pair`: share of photos where two members predict different classes, and where both are wrong. `committee`: mean
pairwise disagreement of its members, the share of photos where every member is wrong, and the share where at
least one member is right (oracle).

| kind | a | b | disagreement | double_fault | oracle |
|---|---|---|---|---|---|
| pair | attention_e1 | attention_e2 | 24.5 | 9.8 | – |
| pair | attention_e1 | attention_e3 | 16.0 | 9.8 | – |
| pair | attention_e1 | inceptionv3_224 | 21.3 | 8.8 | – |
| pair | attention_e1 | efficientnetv2s_224 | 18.8 | 9.0 | – |
| pair | attention_e1 | convnext_tiny_224 | 18.1 | 8.5 | – |
| pair | attention_e1 | swin_tiny_224 | 17.5 | 8.4 | – |
| pair | attention_e1 | convnext_tiny_448 | 16.1 | 7.6 | – |
| pair | attention_e2 | attention_e3 | 23.4 | 11.7 | – |
| pair | attention_e2 | inceptionv3_224 | 26.3 | 11.9 | – |
| pair | attention_e2 | efficientnetv2s_224 | 25.3 | 11.0 | – |
| pair | attention_e2 | convnext_tiny_224 | 26.1 | 10.2 | – |
| pair | attention_e2 | swin_tiny_224 | 23.7 | 10.7 | – |
| pair | attention_e2 | convnext_tiny_448 | 23.7 | 9.2 | – |
| pair | attention_e3 | inceptionv3_224 | 21.8 | 9.8 | – |
| pair | attention_e3 | efficientnetv2s_224 | 19.9 | 9.6 | – |
| pair | attention_e3 | convnext_tiny_224 | 19.9 | 8.9 | – |
| pair | attention_e3 | swin_tiny_224 | 18.8 | 8.9 | – |
| pair | attention_e3 | convnext_tiny_448 | 17.7 | 8.2 | – |
| pair | inceptionv3_224 | efficientnetv2s_224 | 17.4 | 12.3 | – |
| pair | inceptionv3_224 | convnext_tiny_224 | 18.4 | 11.1 | – |
| pair | inceptionv3_224 | swin_tiny_224 | 17.9 | 10.9 | – |
| pair | inceptionv3_224 | convnext_tiny_448 | 17.9 | 9.7 | – |
| pair | efficientnetv2s_224 | convnext_tiny_224 | 16.1 | 11.3 | – |
| pair | efficientnetv2s_224 | swin_tiny_224 | 15.5 | 11.1 | – |
| pair | efficientnetv2s_224 | convnext_tiny_448 | 15.8 | 9.3 | – |
| pair | convnext_tiny_224 | swin_tiny_224 | 15.0 | 10.4 | – |
| pair | convnext_tiny_224 | convnext_tiny_448 | 15.1 | 9.2 | – |
| pair | swin_tiny_224 | convnext_tiny_448 | 15.2 | 8.8 | – |
| committee | row2_plain | convnext_tiny_448+attention_e1 | 16.1 | 7.6 | 92.4 |
| committee | row3_plain | convnext_tiny_448+swin_tiny_224 | 15.2 | 8.8 | 91.2 |

## 6. Test-time augmentation (eight views), best whole-image member

| model | macro-F1 (mean ± SD) | balanced acc. (mean ± SD) |
|---|---|---|
| convnext_tiny_448, one view | 79.2 ± 3.8 | 79.1 ± 5.0 |
| convnext_tiny_448, eight views | 81.3 ± 4.3 | 80.8 ± 5.9 |

| A | B | metric | difference (points) | photo range | block range | rounds won (A) | verdict, photo range | verdict, block range |
|---|---|---|---|---|---|---|---|---|
| eight views | one view | macro_f1 | +2.0 | [+1.1, +2.9] | [+1.1, +3.4] | 4/5 | A beats B | A beats B |
| eight views | one view | balanced_accuracy | +1.7 | [+0.8, +2.6] | [+0.8, +3.0] | 4/5 | A beats B | A beats B |

Cost column: pending: Stage 13.
