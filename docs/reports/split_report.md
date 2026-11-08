# Stage 3 split report

Built by `src/split.py`; do not edit by hand.

Primary dataset: 3334 photos in 3045 near-duplicate groups (21 photos excluded as label conflicts). StratifiedGroupKFold, 5 folds, seed 2026; validation = a stratified, grouped 1/8 of the other four folds (seed 2026 + round).

**`splits/folds.csv` SHA-256:** `9891c146c2a369ac455a4794a4707509a1508b3005a680b5d2257206f7023ba7`

Checks:

- every photo tested exactly once: **yes**
- no group in two roles within a round: **yes**
- class shares within 2% points: **yes**
- validation and test never overlap: **yes**
- worst class-share deviation: **0.21%**

| round | role | photos | healthy | leaf_blast | brown_spot | hispa |
|---|---|---|---|---|---|---|
| 1 | train | 2332 | 1036 | 541 | 361 | 394 |
| 1 | val | 335 | 148 | 78 | 52 | 57 |
| 1 | test | 667 | 296 | 154 | 104 | 113 |
| 2 | train | 2333 | 1036 | 541 | 361 | 395 |
| 2 | val | 334 | 148 | 78 | 52 | 56 |
| 2 | test | 667 | 296 | 154 | 104 | 113 |
| 3 | train | 2334 | 1036 | 541 | 362 | 395 |
| 3 | val | 333 | 148 | 77 | 52 | 56 |
| 3 | test | 667 | 296 | 155 | 103 | 113 |
| 4 | train | 2333 | 1036 | 540 | 362 | 395 |
| 4 | val | 334 | 148 | 78 | 52 | 56 |
| 4 | test | 667 | 296 | 155 | 103 | 113 |
| 5 | train | 2334 | 1036 | 540 | 362 | 396 |
| 5 | val | 334 | 148 | 78 | 52 | 56 |
| 5 | test | 666 | 296 | 155 | 103 | 112 |

Paddy Doctor test list `splits/paddy4.csv`: 8086 photos (healthy 2373, leaf_blast 2332, brown_spot 1240, hispa 2141); each byte-identical image once; SHA-256 `f07ef2a215ef9527a35417654c118c70b5b1b478f864dab933731e07480fe54d`.
