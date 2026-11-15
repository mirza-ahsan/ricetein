# Session-blocked split report

Built by `src/session_split.py`; do not edit by hand. A sensitivity split declared on 2026-11-14 (handbook §24). The locked split is unchanged.

The same 3334 photos as the locked split, grouped into 74 capture blocks: runs of consecutive photos from one camera with no pause longer than 60 s, merged where they share a near-duplicate group. Block sizes: largest 287 photos (8.6% of all), median 24. StratifiedGroupKFold, 5 folds, seed 2026; validation = a stratified, grouped 1/8 of the other four folds.

**`splits/session_blocked/folds.csv` SHA-256:** `28fa6372f49bb5ff1b082c4ff7ae2b2709a6a70c59f6de1f768b15f3e09aa8f8`

Checks (class shares are reported, not enforced: whole blocks limit how even they can be):

- every photo tested exactly once: **yes**
- no group in two roles within a round: **yes**
- class shares within 2% points: **no**
- validation and test never overlap: **yes**
- worst class-share deviation: **26.53%**

| round | role | photos | healthy | leaf_blast | brown_spot | hispa |
|---|---|---|---|---|---|---|
| 1 | train | 2246 | 1006 | 544 | 290 | 406 |
| 1 | val | 350 | 194 | 75 | 34 | 47 |
| 1 | test | 738 | 280 | 154 | 193 | 111 |
| 2 | train | 2232 | 1018 | 543 | 268 | 403 |
| 2 | val | 452 | 137 | 76 | 190 | 49 |
| 2 | test | 650 | 325 | 154 | 59 | 112 |
| 3 | train | 2315 | 1009 | 543 | 364 | 399 |
| 3 | val | 332 | 178 | 76 | 24 | 54 |
| 3 | test | 687 | 293 | 154 | 129 | 111 |
| 4 | train | 2345 | 990 | 540 | 403 | 412 |
| 4 | val | 346 | 189 | 77 | 40 | 40 |
| 4 | test | 643 | 301 | 156 | 74 | 112 |
| 5 | train | 2350 | 1061 | 544 | 362 | 383 |
| 5 | val | 368 | 138 | 74 | 93 | 63 |
| 5 | test | 616 | 281 | 155 | 62 | 118 |

Share of test photos with a training photo from the same camera taken close in time (all five rounds):

| split | within 5 s | within 10 s | within 60 s |
|---|---|---|---|
| locked (`splits/folds.csv`) | 72.0% | 90.2% | 99.6% |
| session-blocked | 0.0% | 0.0% | 0.0% |
