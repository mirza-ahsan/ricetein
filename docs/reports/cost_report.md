# Stage 13: cost and compression

Built by `src/cost.py` from `runs/`; do not edit by hand. Declared 2026-11-21.

**Hardware dependence.** Every time and every memory figure comes from one machine, `halo` (RTX 4060 8 GB, Ryzen 7 7700,
8 threads for torch and onnxruntime), and depends on it. Times are medians over 100 training photos of round 1, one photo
at a time from the file on disk (decoding included, cold-disk reads not), after 5 discarded warm-up photos. GPU: fp16
autocast, 128 tiles per pass; CPU: fp32, 16 tiles per pass. Each method ran in its own process.

## 1. Cost of every main method

Accuracy is **test** macro-F1 (mean over 5 rounds) of the method's existing test runs; times, sizes and memory are measured here.
The tile method describes about 79 tiles per photo, so it is **slower per photo than a whole-image model**. The training
cache is the description cache (all 8 versions) or the resized-photo cache.

| method | encoder params (M, frozen) | decision-maker params (M) | files (MB) | training cache (GB) | GPU s/photo | CPU s/photo | peak GPU (MB) | peak CPU RAM (MB) | test macro-F1 (%) |
|---|---|---|---|---|---|---|---|---|---|
| attention | 85.7 | 0.23 | 328 | 6.1 | 0.310 | 5.36 | 951 | 1482 | 77.9 |
| mean | 85.7 | 0.40 | 328 | 6.1 | 0.299 | 5.37 | 952 | 1470 | 69.1 |
| max | 85.7 | 0.10 | 327 | 6.1 | 0.303 | 5.42 | 951 | 1469 | 73.4 |
| probe 224 | 85.7 | 0.00 | 327 | 0.2 | 0.092 | 0.15 | 347 | 1317 | 73.0 |
| probe 512 | 85.7 | 0.00 | 327 | 0.2 | 0.103 | 0.42 | 370 | 1321 | 73.5 |
| inceptionv3 224 | 21.8 | 0.01 | 83 | 0.6 | 0.048 | 0.06 | 100 | 1004 | 72.0 |
| efficientnetv2s 224 | 20.2 | 0.01 | 78 | 0.6 | 0.050 | 0.06 | 93 | 1019 | 74.1 |
| convnext tiny 224 | 27.8 | 0.00 | 106 | 0.6 | 0.045 | 0.07 | 128 | 1020 | 75.0 |
| swin tiny 224 | 27.5 | 0.00 | 105 | 0.6 | 0.047 | 0.07 | 130 | 1020 | 76.3 |
| convnext tiny 448 | 27.8 | 0.00 | 106 | 2.5 | 0.047 | 0.12 | 151 | 1020 | 79.2 |
| committee | 113.5 | 0.23 | 434 | 8.5 | 0.357 | 5.47 | 951 | 1482 | 81.6 |
| final small model (e1 top16) | 85.7 | 0.23 | 328 | 6.1 | 0.138 | 1.21 | 435 | 1371 | 76.7 |

## 2. Compression variants (validation macro-F1)

Validation photos only (session split, mean ± SD over 5 rounds); no variant was tested except the final small model.
The round-models are applied as they are (nothing retrained, nothing tuned), except the distilled head.

| variant | what | validation macro-F1 (%) | difference (points) | tiles per photo | GPU s/photo | CPU s/photo | note |
|---|---|---|---|---|---|---|---|
| attention | E1 + attention, all tiles (the reference) | 78.5 ± 5.7 | +0.0 | 80 | 0.310 | 5.36 |  |
| e1_top16 | e1 top16 | 76.5 ± 5.9 | -2.0 | 16 | 0.138 | 1.21 |  |
| e1_even16 | e1 even16 | 49.0 ± 7.5 | -29.5 | 16 | – | – | control (not a candidate) |
| e4 | e4 | 74.8 ± 7.9 | -3.7 | 80 | 0.172 | 1.66 |  |
| e4_top16 | e4 top16 | 73.7 ± 8.7 | -4.8 | 16 | 0.110 | 0.41 |  |
| e4_even16 | e4 even16 | 49.5 ± 7.9 | -29.0 | 16 | – | – | control (not a candidate) |
| distilled_e4 | distilled e4 | 78.6 ± 7.3 | +0.1 | 80 | 0.172 | 1.66 |  |
| distilled_e4_top16 | distilled e4 top16 | 76.1 ± 9.0 | -2.4 | 16 | 0.110 | 0.41 |  |
| attention_int8 | E1 with 8-bit linear layers, on a subset of 399 validation photos | 23.6 ± 4.4 | -54.0 | – | – | 3.55 | difference is against the float descriptions of the same subset (77.6%); mean cosine of float and 8-bit tile descriptions 0.39 |
| final_onnx | e1_top16 exported to ONNX (onnxruntime, CPU) | 76.5 ± 5.9 | -2.0 | 16 | – | 0.93 | same model, other runtime |

## 3. The final small model: e1_top16

Chosen on validation by the declared rule (`runs/cost_selection.json`), frozen in `runs/cost_plan.json`, tested once
(`runs/cost_final/test_started.json`). Test, session split. Attention is A and the small model is B, so a positive
difference means attention is better.

| metric | difference A - B (points) | photo range | block range | rounds won by attention | verdict (photo range) | verdict (block range) |
|---|---|---|---|---|---|---|
| macro_f1 | +1.4 | [0.6, 2.1] | [0.5, 2.5] | 3/5 | no clear difference | no clear difference |
| balanced_accuracy | +0.8 | [0.2, 1.5] | [-0.0, 1.9] | 3/5 | no clear difference | no clear difference |

**Limits.** One machine; one seed per round; the 8-bit row rests on a subset of validation photos; the distillation teacher
answers on its own training photos, so its answers are close to the labels; the tile rule is a simple colour heuristic.
