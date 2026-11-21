# Core study results (locked split)

Original tuning/epoch budgets and five rounds; seed-0 headline models only. Test was evaluated once after validation selection. Stability seeds, 128/Otsu/448 whole-image settings, ablations, partition-order and Paddy evaluation are deferred. Incomplete whole-image searches are excluded rather than selected from a reduced budget: []. Scores below are percentages; ECE is a proportion. Confidence intervals use class-stratified pooled out-of-fold bootstrap resampling. Paired differences are percentage points (pp).

| method | macro_f1 mean ± SD | balanced_accuracy mean ± SD | accuracy mean ± SD | pooled macro_f1 [95% CI] | pooled balanced_accuracy [95% CI] | hispa_recall | ece |
|---|---|---|---|---|---|---|---|
| inceptionv3_224 | 80.9 ± 2.7 | 81.7 ± 2.2 | 82.2 ± 2.6 | 80.9 [79.5, 82.3] | 81.7 [80.4, 83.2] | 73.8 | 0.049 |
| efficientnetv2s_224 | 82.8 ± 1.0 | 82.9 ± 1.0 | 84.1 ± 1.1 | 82.8 [81.4, 84.1] | 82.9 [81.5, 84.2] | 72.7 | 0.058 |
| convnext_tiny_224 | 83.4 ± 2.6 | 83.5 ± 2.9 | 84.7 ± 2.2 | 83.4 [82.1, 84.7] | 83.5 [82.2, 85.0] | 72.7 | 0.052 |
| swin_tiny_224 | 83.7 ± 1.3 | 84.5 ± 1.9 | 84.7 ± 1.2 | 83.8 [82.5, 85.0] | 84.5 [83.2, 85.8] | 78.0 | 0.060 |
| probe_224 | 78.2 ± 0.7 | 79.0 ± 0.5 | 79.8 ± 1.2 | 78.3 [76.8, 79.7] | 79.0 [77.4, 80.5] | 70.9 | 0.024 |
| probe_512 | 81.3 ± 1.1 | 82.0 ± 1.1 | 82.7 ± 1.0 | 81.3 [79.9, 82.6] | 82.0 [80.6, 83.4] | 76.2 | 0.020 |
| mean | 76.0 ± 1.5 | 76.4 ± 1.6 | 77.6 ± 1.6 | 76.0 [74.5, 77.6] | 76.4 [74.9, 78.0] | 69.0 | 0.087 |
| max | 82.9 ± 1.7 | 83.4 ± 1.7 | 84.0 ± 1.7 | 82.9 [81.5, 84.2] | 83.4 [82.1, 84.8] | 73.6 | 0.122 |
| attention | 82.4 ± 2.0 | 83.2 ± 2.2 | 83.3 ± 2.0 | 82.4 [81.0, 83.6] | 83.2 [81.9, 84.6] | 77.5 | 0.089 |

| A | B | metric | difference (pp) | 95% CI (pp) | round wins | verdict |
|---|---|---|---|---|---|---|
| attention | inceptionv3_224 | macro_f1 | 1.5 | [-0.1, 3.0] | 2/5 | no clear difference |
| attention | inceptionv3_224 | balanced_accuracy | 1.5 | [-0.1, 3.0] | 2/5 | no clear difference |
| attention | efficientnetv2s_224 | macro_f1 | -0.4 | [-2.0, 1.1] | 2/5 | no clear difference |
| attention | efficientnetv2s_224 | balanced_accuracy | 0.4 | [-1.2, 1.9] | 4/5 | no clear difference |
| attention | convnext_tiny_224 | macro_f1 | -1.0 | [-2.6, 0.5] | 2/5 | no clear difference |
| attention | convnext_tiny_224 | balanced_accuracy | -0.3 | [-1.9, 1.3] | 3/5 | no clear difference |
| attention | swin_tiny_224 | macro_f1 | -1.4 | [-2.9, 0.0] | 2/5 | no clear difference |
| attention | swin_tiny_224 | balanced_accuracy | -1.2 | [-2.7, 0.2] | 1/5 | no clear difference |
| attention | probe_224 | macro_f1 | 4.1 | [2.4, 5.6] | 5/5 | A beats B |
| attention | probe_224 | balanced_accuracy | 4.3 | [2.6, 5.9] | 5/5 | A beats B |
| attention | probe_512 | macro_f1 | 1.1 | [-0.4, 2.5] | 3/5 | no clear difference |
| attention | probe_512 | balanced_accuracy | 1.2 | [-0.2, 2.7] | 4/5 | no clear difference |
| attention | mean | macro_f1 | 6.3 | [5.0, 7.8] | 5/5 | A beats B |
| attention | mean | balanced_accuracy | 6.8 | [5.4, 8.2] | 5/5 | A beats B |
| attention | max | macro_f1 | -0.5 | [-1.6, 0.6] | 1/5 | no clear difference |
| attention | max | balanced_accuracy | -0.2 | [-1.3, 1.0] | 1/5 | no clear difference |
