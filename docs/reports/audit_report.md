# Stage 2 audit report

Built by `src/audit_report.py` from the audit files; do not edit by hand.

## 1. Backgrounds (2a)

Sample: 60 random photos per class (seed 0), frozen in `data/background_audit_sample.csv`. Two raters label every photo alone, in their own shuffled order, without seeing its class: plain or cluttered, plus several leaves, hand, strong shadow. Disagreements on the background are settled together; both original labels are kept.

**Plain background: 100.0%** of 240 photos (95% CI 98.5%-100.0%). Raters model_rater_a and model_rater_b agree on 99.6% of photos (Cohen's kappa 0.00); 1 disagreements were settled together. Model raters, not people: `model_rater_a`, `model_rater_b` (independent instances of one AI model; initial reviews did not use each other's answers. Model agreement does not establish human inter-rater reliability). File: `data/background_audit.csv`.

| class | photos | plain | 95% CI |
|---|---|---|---|
| brown_spot | 60 | 100.0% | 94.0%-100.0% |
| healthy | 60 | 100.0% | 94.0%-100.0% |
| hispa | 60 | 100.0% | 94.0%-100.0% |
| leaf_blast | 60 | 100.0% | 94.0%-100.0% |

Flags (share of photos, per rater):

| flag | model_rater_a | model_rater_b |
|---|---|---|
| multiple_leaves | 0.0% | 0.0% |
| hand | 0.0% | 0.4% |
| strong_shadow | 15.8% | 21.2% |

## 2. Near-duplicates (2b) and overlap with Paddy Doctor (2c)

Flag rules (pre-declared methodology): perceptual hashes within 8 bits (any of 8 rotations/flips), or whole-photo E1 class-token cosine > 0.95. Automatic check: SIFT keypoints (photos at 1200 px), ratio test 0.75, each keypoint used once, RANSAC similarity transform (tolerance 8.0 px, scale 0.25-4.0x, also against the mirrored photo); a pair passes with >= 15 distinct inliers. Raters review every passed pair plus 200 random rejected pairs per comparison, blind to class and check result (deviation from "review every flagged pair": handbook §24, 2026-11-07).

| comparison | flagged | hash only | cosine only | both | pass check | queued |
|---|---|---|---|---|---|---|
| primary x Paddy Doctor | 2 | 2 | 0 | 0 | 0 | 2 |
| within primary | 84978 | 84157 | 724 | 97 | 373 | 573 |

Files: `data/duplicate_candidates.csv`, `data/duplicate_review_queue.csv`.

Reviewers model_rater_b and model_rater_c: Cohen's kappa 0.77; 61 disagreements adjudicated. Duplicate disagreements were adjudicated by model_rater_c with Ahsan's approval; original verdicts are preserved. Protocol and high-resolution rechecks: `data/duplicate_review/adjudication_protocol.json`. Model raters, not people: `model_rater_b`, `model_rater_c` (AI model instances; initial reviews did not use each other's answers. Model agreement does not establish human inter-rater reliability).

| comparison | round | passed | reviewed | confirmed same leaf |
|---|---|---|---|---|
| cross | 1 | False | 2 | 0 |
| primary | 1 | False | 200 | 0 |
| primary | 1 | True | 373 | 349 |

What the automatic check misses: primary x Paddy Doctor: 0 of 2 sampled rejected pairs confirmed (miss rate 0.0%, 95% CI 0.0%-84.2%); within primary: 0 of 200 sampled rejected pairs confirmed (miss rate 0.0%, 95% CI 0.0%-1.8%).

**Groups:** 3054 groups for 3355 photos; group sizes 1: 2840, 2: 170, 3: 24, 4: 12, 5: 1, 6: 4, 7: 1, 8: 1, 11: 1. **Label conflicts:** 21 photos in confirmed pairs with different classes (excluded from all experiments; `data/label_conflicts.csv`). **Overlap with Paddy Doctor:** 0 confirmed matches (`data/cross_dataset_overlap.csv`).

## 3. Label sanity check (2d)

Each configured checker looks at 20 random photos per class with the label shown, and says whether it looks right, looks like another class, or is unclear. No label is changed: the main results always use the original labels.

Checkers: model_rater_a. Model raters, not people: `model_rater_a` (independent instances of one AI model; initial reviews did not use each other's answers. Model agreement does not establish human inter-rater reliability). 80 answers on 80 photos; 1 answers on 1 photos look like another class (`data/suspect_labels.csv`).

| class | looks_like | answers |
|---|---|---|
| healthy | hispa | 1 |
