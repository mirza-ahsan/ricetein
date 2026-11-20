# Stage 10: Paddy Doctor zero-shot test, pre-declaration

Written 2026-11-20, **before** any Paddy Doctor photo was described or scored. It is frozen when it is committed;
a later change is a new dated declaration, never an edit of this file. The same plan as a settings file is
`configs/paddy_zero_shot.yaml`.

**Question (H3).** Do features and models learned only on the primary dataset still work on field photos they
have never seen? Paddy Doctor is never used for training, tuning, selection or threshold setting. It is scored once.

## 1. The photos

| Item | Value |
|---|---|
| File | `splits/paddy4.csv`: 8,086 photos of the four tested classes (healthy = `normal`, `blast`, `brown_spot`, `hispa`), 1080 × 1440 px; 78 byte-identical extra copies are already removed |
| Class order | `[healthy, leaf_blast, brown_spot, hispa]`, fixed |
| Not used | the nine other Paddy Doctor classes, except for the leaf-width measurement below |

## 2. The zoom problem and the three zoom levels

A 224 px tile covers a different amount of real leaf in the two datasets. The primary photos are brought to a
common leaf scale (median leaf width 194.9 px, `data/pixel_scale_decision.json`). To compare, leaf width was
measured on **30 Paddy Doctor photos from the nine classes we do not test** (3 per class and 3 extra, seeded
draw; `data/paddy_leaf_width_measurements.csv`).

- **Who measured:** a model rater (`model_rater_a`), by eye against a 200 px grid, one typical in-focus leaf blade per photo.
  There are **no click coordinates**, and no person measured these photos. This is disclosed wherever the factor is used.
- **Result:** median leaf width 64.5 px (bootstrap 95% range of the median 57.5 to 72.5). The widths run from
  about 20 px (distant field shots) to about 125 px (close-ups).
- **Matching factor:** 194.9 / 64.5 = 3.02 (range 2.69 to 3.39), **rounded to 3.0**.

| Zoom level | Photo scale before tiling | Tiles per photo (about) |
|---|---|---|
| **native (headline)** | 1.0 (the photo as stored) | 35 |
| midway | 1.73 (square root of 3.0) | 108 |
| matched | 3.0 | 300 |

**The headline is native**, because it needs no dataset-specific adjustment. This is fixed now: the headline stays
native even if another level scores better. Other levels are secondary results.

## 3. What is scored

Every method is scored with **each of its five round-models** (rounds 1 to 5 of the session-blocked split).

| Group | Methods | Models |
|---|---|---|
| Headline methods (`runs/session_selection.json`) | InceptionV3, EfficientNetV2-S, ConvNeXt-T, Swin-T (224 px), probe 224, probe 512, mean, max, attention, `whole224`, `whole512` | 11 × 5 = 55 |
| Stage 5 conditions (`runs/session_conditions_selection.json`) | the four models at 128 px and at 224 px with Otsu, and the best model at 448 px | 9 × 5 = 45, scored once each when their training is complete |

Models trained on the locked split are not scored: the locked split is not the headline split.

**Input of each family** (always the clean photo, no augmentation):

| Family | Input |
|---|---|
| mean, max, attention | the frozen E1 tile descriptions at each of the three zoom levels (tile grid rule of Stage 6.1, 224 px tiles, no Otsu) |
| probe 224, probe 512, `whole224`, `whole512` | the frozen E1 whole-photo description: the photo padded with black to a square, then resized to 224 or 512 px. No zoom levels |
| whole-image models | the photo padded with black to a square, resized to the model's cache side (⌊8s/7⌋) and then to s. For the Otsu conditions, Otsu cleaning is applied to the **unpadded** photo, then the photo is padded. No zoom levels |

## 4. What is reported

1. Macro-F1, balanced accuracy, per-class recall and the confusion matrix, for every method, as mean ± SD over the
   five round-models, and for the **5-model committee** (the average of the five probability vectors). A committee is
   legitimate here because none of the five models saw Paddy Doctor.
2. For each tile method, all three zoom levels, with the headline marked.
3. The **drop**: the method's session-split test macro-F1 (`docs/reports/session_report.md`) minus its Paddy Doctor
   macro-F1, per round-model and on average. The drop is reported for every method.
4. Attention against every other method at the headline zoom, with the decision rule: the paired photo
   bootstrap (2,000 resamples, 95%, stratified, `base.yaml`) of the difference between the two committees must be above 0
   **and** attention must win at least 4 of the 5 rounds (round-model against round-model). A block bootstrap is not
   possible here: capture blocks are not known for Paddy Doctor photos, so only the photo range is given.
5. **H3 reading rule:** H3 is supported if attention at the native zoom beats each fine-tuned whole-image model at
   224 px under item 4. If not, we report that as it is. Drops are described, not used for a claim on their own.

## 5. Rules

- **Run once.** Each (model, zoom level) is scored once. No tuning, no threshold change, no second try, no choice
  of zoom level after the fact.
- **A bug found afterwards:** fix it, log it, and rerun every affected configuration (the bug-fix rule).
- **No Paddy Doctor photo of the four tested classes was looked at** while writing this plan. The 30 measured photos
  are from the nine untested classes only.

## 6. Limits

- The zoom factor rests on 30 visual estimates by one model rater; the single number 3.0 hides a wide spread of
  photo scales. The headline (native) does not depend on it.
- The models were chosen on session-split validation photos, which is a different dataset from Paddy Doctor.
  Chance overlap of the two datasets is limited by the cross-dataset overlap list (`data/cross_dataset_overlap.csv`).
- Whole-image models see one resized, padded photo; tile methods see a zoomed tile grid. The comparison is of
  complete pipelines, not of one factor.
