# Stage 5 Otsu quality check

A frozen random sample of 50 primary photos (seed 0; `data/otsu_review_sample.csv`) was inspected side by side using the actual Stage 5 preparation code. **model_rater_c is an AI model rater**, not a human segmentation annotator. Original and cleaned previews are in the ignored `cache/otsu_review/` directory; per-photo judgments are in `data/otsu_review.csv`.

A failure means visible leaf/pale symptom material was removed, or a conspicuous connected background shadow was retained. Thin boundary halos alone were not counted. These are qualitative model judgments, not pixel-level segmentation ground truth. The single batch export timestamp follows inspection; it is not a per-photo decision time.

Failures: **23/50 (46.0%; exact 95% CI 31.8%–60.7%)**: 5 with removed leaf material and 18 with a retained shadow. The fixed preprocessing recipe was not changed in response to this check. No disease-classification test scores were inspected.
