# Stage 2 audit report

Built by `src/audit_report.py` from the audit files; do not edit by hand.

## 1. Backgrounds (2a)

Sample: 60 random photos per class (seed 0), frozen in `data/background_audit_sample.csv`. Two team members label every photo alone, in their own shuffled order, without seeing its class: plain or cluttered, plus several leaves, hand, strong shadow. Disagreements on the background are settled together; both original labels are kept.

**Pending.** Done so far: nobody yet of 240.

## 2. Near-duplicates (2b) and overlap with Paddy Doctor (2c)

Flag rules (pre-declared methodology): perceptual hashes within 8 bits (any of 8 rotations/flips), or whole-photo E1 class-token cosine > 0.95. Automatic check: SIFT keypoints (photos at 1200 px), ratio test 0.75, each keypoint used once, RANSAC similarity transform (tolerance 8.0 px, scale 0.25-4.0x, also against the mirrored photo); a pair passes with >= 15 distinct inliers. People review every passed pair plus 200 random rejected pairs per comparison, blind to class and check result (deviation from "review every flagged pair": handbook §24, 2026-11-07).

| comparison | flagged | hash only | cosine only | both | pass check | queued |
|---|---|---|---|---|---|---|
| primary x Paddy Doctor | 2 | 2 | 0 | 0 | 0 | 2 |
| within primary | 84978 | 84157 | 724 | 97 | 373 | 573 |

Files: `data/duplicate_candidates.csv`, `data/duplicate_review_queue.csv`.

**Pending.** Done so far: nobody yet of 575.

## 3. Label sanity check (2d)

Each team member looks at their own 20 random photos per class with the label shown, and says whether it looks right, looks like another class, or is unclear. No label is changed: the main results always use the original labels.

**Pending.** Done so far: nobody yet of 80 each.
