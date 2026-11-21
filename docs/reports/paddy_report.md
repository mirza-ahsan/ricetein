# Paddy Doctor zero-shot test

Built by `src/paddy_zero_shot.py` from `runs/*/paddy_predictions*.csv`; do not edit by hand. Plan: `docs/reports/paddy_pre_declaration.md` (frozen before the run). All scores are percentages on the 8086 Paddy Doctor photos of the four tested classes; nothing was trained or tuned on them. Per-round scores, per-class recall of every round and the confusion matrices are in `results/tables/paddy_rounds.csv` and `paddy_confusion.csv`.

## 1. Results (mean ± SD over the five round-models, and the 5-model committee)

| method | zoom | macro-F1 | balanced acc. | committee macro-F1 | committee recall healthy | committee recall leaf_blast | committee recall brown_spot | committee recall hispa | session test macro-F1 | drop |
|---|---|---|---|---|---|---|---|---|---|---|
| inceptionv3_224 | - | 16.6 ± 6.6 | 27.9 ± 3.7 | 11.6 | 0.0 | 100.0 | 0.0 | 0.8 | 72.0 | 55.4 ± 6.2 |
| efficientnetv2s_224 | - | 27.0 ± 5.5 | 32.5 ± 5.2 | 29.4 | 48.6 | 90.4 | 3.1 | 4.3 | 74.1 | 47.2 ± 8.5 |
| convnext_tiny_224 | - | 30.1 ± 7.3 | 35.5 ± 4.8 | 29.8 | 18.5 | 84.7 | 27.4 | 8.9 | 75.0 | 44.9 ± 9.2 |
| swin_tiny_224 | - | 27.0 ± 5.8 | 32.6 ± 2.6 | 25.0 | 21.0 | 96.3 | 11.8 | 1.0 | 76.3 | 49.4 ± 8.2 |
| probe_224 | - | 26.1 ± 4.1 | 29.3 ± 2.1 | 27.7 | 15.6 | 57.2 | 25.3 | 19.7 | 73.0 | 46.9 ± 7.4 |
| probe_512 | - | 26.1 ± 6.5 | 30.0 ± 4.8 | 29.6 | 21.2 | 69.8 | 7.7 | 28.3 | 73.5 | 47.4 ± 12.0 |
| mean | native (headline) | 16.7 ± 2.4 | 27.7 ± 1.5 | 15.2 | 0.0 | 99.5 | 0.0 | 8.6 | 69.1 | 52.4 ± 7.6 |
| mean | midway | 14.8 ± 1.0 | 26.6 ± 0.5 | 13.6 | 0.0 | 99.8 | 0.0 | 4.8 | 69.1 | 54.4 ± 7.8 |
| mean | matched | 15.2 ± 1.3 | 26.7 ± 0.6 | 14.3 | 0.0 | 99.5 | 0.3 | 6.0 | 69.1 | 54.0 ± 7.8 |
| max | native (headline) | 20.7 ± 3.8 | 29.0 ± 1.6 | 21.0 | 0.0 | 93.6 | 7.1 | 18.4 | 73.4 | 52.7 ± 4.6 |
| max | midway | 17.5 ± 4.3 | 27.5 ± 1.8 | 15.5 | 0.0 | 97.7 | 1.0 | 8.5 | 73.4 | 55.9 ± 6.1 |
| max | matched | 16.8 ± 4.9 | 27.2 ± 2.1 | 14.7 | 0.0 | 98.8 | 0.5 | 6.8 | 73.4 | 56.6 ± 7.3 |
| attention | native (headline) | 26.1 ± 4.2 | 33.8 ± 2.9 | 25.5 | 0.1 | 79.2 | 37.9 | 15.8 | 77.9 | 51.9 ± 3.5 |
| attention | midway | 21.5 ± 4.0 | 30.5 ± 2.6 | 19.8 | 0.0 | 90.0 | 20.6 | 6.4 | 77.9 | 56.5 ± 3.6 |
| attention | matched | 17.7 ± 2.9 | 28.1 ± 1.5 | 16.2 | 0.0 | 96.9 | 8.6 | 3.4 | 77.9 | 60.3 ± 2.1 |
| whole224 | - | 25.4 ± 4.2 | 30.7 ± 2.5 | 24.9 | 6.7 | 19.9 | 63.5 | 30.5 | 69.3 | 43.9 ± 10.6 |
| whole512 | - | 24.0 ± 6.4 | 32.9 ± 2.1 | 25.3 | 2.9 | 35.2 | 74.1 | 20.3 | 73.2 | 49.2 ± 12.8 |
| inceptionv3_128 | - | 18.7 ± 4.9 | 28.8 ± 2.9 | 13.8 | 1.0 | 99.1 | 4.4 | 0.0 | 56.9 | 38.2 ± 9.7 |
| inceptionv3_224_otsu | - | 13.5 ± 4.2 | 25.3 ± 2.5 | 14.4 | 0.0 | 38.7 | 58.7 | 0.0 | 71.3 | 57.7 ± 4.7 |
| efficientnetv2s_128 | - | 25.7 ± 2.2 | 32.9 ± 0.9 | 32.4 | 33.2 | 87.8 | 0.1 | 28.4 | 62.8 | 37.1 ± 6.6 |
| efficientnetv2s_224_otsu | - | 17.2 ± 7.2 | 25.3 ± 2.5 | 13.4 | 0.0 | 71.7 | 16.5 | 0.7 | 70.2 | 53.0 ± 6.0 |
| convnext_tiny_128 | - | 29.8 ± 4.9 | 33.5 ± 3.4 | 30.0 | 40.8 | 53.8 | 39.8 | 2.9 | 67.8 | 38.0 ± 6.2 |
| convnext_tiny_224_otsu | - | 18.1 ± 4.1 | 25.8 ± 2.8 | 16.4 | 2.2 | 65.5 | 31.3 | 0.6 | 73.5 | 55.4 ± 4.2 |
| swin_tiny_128 | - | 20.1 ± 5.6 | 28.5 ± 2.8 | 18.0 | 1.3 | 92.6 | 14.6 | 2.7 | 65.2 | 45.1 ± 8.1 |
| swin_tiny_224_otsu | - | 21.2 ± 4.7 | 27.8 ± 3.7 | 18.5 | 0.7 | 85.8 | 8.0 | 11.8 | 71.7 | 50.5 ± 7.8 |
| convnext_tiny_448 | - | 26.2 ± 7.7 | 32.7 ± 4.8 | 22.7 | 5.1 | 95.9 | 7.0 | 14.0 | 79.2 | 53.0 ± 10.2 |

## 2. Attention at the headline zoom (native) against every other method (committees, photo bootstrap)

| B | difference (points) | photo range | rounds won | verdict |
|---|---|---|---|---|
| inceptionv3_224 | 13.9 | [13.2, 14.7] | 5/5 | A beats B |
| efficientnetv2s_224 | -3.8 | [-4.9, -2.8] | 2/5 | no clear difference |
| convnext_tiny_224 | -4.3 | [-5.4, -3.1] | 1/5 | no clear difference |
| swin_tiny_224 | 0.5 | [-0.6, 1.6] | 3/5 | no clear difference |
| probe_224 | -2.1 | [-3.3, -1.0] | 2/5 | no clear difference |
| probe_512 | -4.1 | [-5.2, -3.0] | 2/5 | no clear difference |
| mean | 10.3 | [9.5, 11.1] | 5/5 | A beats B |
| max | 4.6 | [3.8, 5.3] | 4/5 | A beats B |
| whole224 | 0.7 | [-0.5, 1.7] | 3/5 | no clear difference |
| whole512 | 0.2 | [-0.7, 1.2] | 3/5 | no clear difference |
| inceptionv3_128 | 11.7 | [10.9, 12.6] | 4/5 | A beats B |
| inceptionv3_224_otsu | 11.1 | [10.3, 12.0] | 5/5 | A beats B |
| efficientnetv2s_128 | -6.8 | [-7.9, -5.8] | 2/5 | no clear difference |
| efficientnetv2s_224_otsu | 12.1 | [11.2, 13.0] | 5/5 | A beats B |
| convnext_tiny_128 | -4.4 | [-5.6, -3.4] | 1/5 | no clear difference |
| convnext_tiny_224_otsu | 9.2 | [8.2, 10.1] | 5/5 | A beats B |
| swin_tiny_128 | 7.6 | [6.6, 8.5] | 4/5 | A beats B |
| swin_tiny_224_otsu | 7.1 | [6.1, 8.1] | 4/5 | A beats B |
| convnext_tiny_448 | 2.9 | [1.8, 4.0] | 2/5 | no clear difference |

**H3 reading rule (pre-declaration, item 5):** attention at the native zoom beats each of the 4 fine-tuned 224 px models: no, H3 is not shown by this rule.
