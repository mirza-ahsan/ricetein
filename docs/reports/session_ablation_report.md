# Stage 8 ablations on the session-blocked split

Built by `src/session_ablations.py` from `runs/`; do not edit by hand. Every variant changes one thing and uses the settings selected for the main configuration (no tuning for any variant, declared 2026-11-20). Five rounds, one seed.

**Limits.** Validation scores favour the main configuration: its settings were chosen on these same validation photos, the variants' were not. The settings were also tuned for the E1 class-token plus patch-average description (1,536 numbers), so a variant on another encoder or description may be handicapped (E4 has 384 dimensions; E2 gives one pooled vector, so its variant changes the encoder and the description together). The row "main, seeds 1 and 2" shows how much the same configuration moves with the training seed.

## 1. Validation macro-F1 (main and all trained variants)

| variant | what changes | validation macro-F1 | difference vs main (points) | rounds better than main |
|---|---|---|---|---|
| main | E1 + attention, selected settings | 78.5 ± 5.7 | +0.0 | – |
| main, seeds 1 and 2 | The same configuration, other training seeds (noise floor) | 77.1 ± 5.9 | -1.4 | 1 |
| otsu_on | Tiles of Otsu-cleaned photos | 74.5 ± 7.6 | -4.0 | 0 |
| native_scale | Native resolution instead of the common leaf scale | 75.0 ± 7.1 | -3.5 | 1 |
| tile448 | 448 px tiles shrunk to 224 px | 77.8 ± 4.5 | -0.7 | 2 |
| encoder_e2 | Supervised ConvNeXt-T encoder (pooled output only, so the class-token description) | 66.5 ± 6.9 | -12.0 | 0 |
| encoder_e3 | DINOv2-B encoder | 76.1 ± 6.3 | -2.4 | 1 |
| encoder_e4 | DINOv3-S encoder (384 dimensions) | 74.8 ± 7.9 | -3.7 | 1 |
| unweighted_loss | Unweighted loss | 77.9 ± 6.4 | -0.6 | 1 |
| single_version | Clean photos only (no cached augmentation) | 75.2 ± 6.8 | -3.3 | 0 |
| tile_dropout_on | Tile dropout 0.15 (the selected setting is 0) | 77.1 ± 5.3 | -1.4 | 0 |
| description_c | Class token only | 77.5 ± 6.2 | -1.0 | 0 |
| description_m | Patch average only | 74.8 ± 6.0 | -3.7 | 0 |

## 2. Headline ablations on test (main against variant)

| variant | metric | difference (points) | photo range | block range | rounds won by main | reading (photo range) | verdict (block range) |
|---|---|---|---|---|---|---|---|
| otsu_on | macro_f1 | 5.2 | [3.8, 6.7] | [3.3, 7.3] | 5/5 | main beats variant | main beats variant |
| otsu_on | balanced_accuracy | 4.1 | [2.7, 5.6] | [2.2, 6.4] | 5/5 | main beats variant | main beats variant |
| native_scale | macro_f1 | 2.9 | [1.5, 4.4] | [1.0, 5.5] | 4/5 | main beats variant | main beats variant |
| native_scale | balanced_accuracy | 2.3 | [0.9, 3.8] | [0.3, 5.1] | 4/5 | main beats variant | main beats variant |
