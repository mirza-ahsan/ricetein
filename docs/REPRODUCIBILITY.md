# Reproducibility

One page. It points to the dataset versions and download dates, the split files, the background audit and the re-derived counts, says how to rebuild the tables, figures, paper and notebook from `runs/`, and says what is not included and why.
It only points to files that exist in this repository.

## 1. What is published, and where

| Item | Where | Notes |
|---|---|---|
| Dataset versions and download dates | `data/DATASETS.md` | Source, Kaggle ID, download date, archive size and SHA-256, licence (primary: MIT; Paddy Doctor: CC BY 4.0), pinned weight commits |
| Per-photo inventory | `data/inventory.csv`, `data/paddy_inventory.csv`, `docs/reports/inventory_report.md` | SHA-256, perceptual hash, size, camera, capture time and JPEG quality of every photo |
| Re-derived counts | `docs/reports/inventory_report.md`, `data/DATASETS.md` §1–2 | 3,355 photos in the primary download (the published count is 3,353), 3,334 eligible photos; class counts, size bands and the 265,349 tiles at the common leaf scale (330,618 at native resolution) |
| Split files | `splits/folds.csv` (locked split), `splits/session_blocked/folds.csv` (headline split), `splits/paddy4.csv`, `splits/partition_order/` (three deliberately faulty set-ups of the partition-order experiment) | Made once and never regenerated. Checksums: `docs/reports/split_report.md` and `docs/reports/session_split_report.md` |
| Split reports | `docs/reports/split_report.md`, `docs/reports/session_split_report.md` | Fold sizes, class shares, the near-duplicate groups and capture blocks kept together |
| Near-duplicate groups | `data/groups.csv`, `data/duplicate_pairs.csv`, `data/duplicate_candidates.csv`, `data/duplicate_review/` | Review by disclosed AI model raters |
| Background audit | `data/background_audit.csv`, `data/background_audit/`, `docs/reports/audit_report.md` | Two disclosed AI model raters |
| Label check | `data/label_check/`, `data/label_conflicts.csv`, `data/suspect_labels.csv` | Disclosed AI model raters |
| Leaf width and the common leaf scale | `data/leaf_width_measurements.csv`, `data/leaf_width_selection.csv`, `data/pixel_scale_decision.json` | `rescale_factor()` in `src/leaf_width.py` |
| Run records | `runs/<run>/` (config, commit, split checksum, data version, per-photo predictions; no checkpoints) | Every number in the paper traces to one |
| Commit map | `docs/commit_map.csv` | Run records cite the commit ids of the history before its public release; this file maps each of them to its commit in this repository |
| Reports | `docs/reports/` | One per stage, generated from `runs/` |
| Final tables and figures | `results/tables/final_*.csv|tex`, `results/figures/final_*`, `results/tables/final_provenance.json`, `results/figures/final_figures.json` | Each lists the source files and their SHA-256 |
| Declarations made before the tests | `docs/reports/paddy_pre_declaration.md`, `runs/*_plan.json`, `runs/*_declared.json`; each report names the date its plan was declared | Comparisons declared after the means were seen are marked post-hoc |

## 2. Rebuilding from `runs/`

Everything below runs from the repository root. `uv sync` installs the locked environment (Python 3.12.13). No command below trains, splits or evaluates on test.

| What | Command |
|---|---|
| Tests | `uv run --locked pytest -q` |
| Frozen files unchanged (since the methodology code was frozen) | `git diff --stat e58a219 -- src/augment.py src/config.py src/embed.py src/encoders.py src/inventory.py src/inventory_report.py src/kaggle_bootstrap.py src/leaf_width.py src/linear_probe.py src/metrics.py src/mil_models.py src/otsu.py src/photo_dataset.py src/review_tool.py src/sanity_baselines.py src/seeding.py src/split.py src/stats.py src/tiling.py src/train_mil.py src/train_wholeimage.py pyproject.toml uv.lock` (prints nothing) |
| Final tables | `uv run --locked python -m src.make_tables` |
| Final figures | `uv run --locked python -m src.make_figures` |
| Readiness of every promised table and figure | `uv run --locked python -m src.make_tables --check` (nothing is pending) |
| Copy tables and figures into the paper | `./scripts/sync_paper_assets.sh` (copies unchanged) |
| Every decimal number of the paper and its source | `python3 scripts/check_paper_numbers.py` |
| The paper | `overleaf-pdf paper/paper.tex` (Docker TeX Live image and the root `LatexMk`; output `paper/build/output.pdf`), or upload `paper/` to Overleaf with `paper.tex` as the main file (pdfLaTeX and biber) |
| The notebook | `uv run --locked python -m src.build_notebook`, then execute it: `RICETEIN_PROJECT_ROOT=$PWD uv run --locked jupyter nbconvert --to notebook --execute --inplace notebooks/ricetein.ipynb` (default `RUN_STAGES = False`: it runs in seconds and shows the saved results) |

A rebuild of the tables from the committed `runs/` gives the committed tables again (checked 2026-11-22; only the code-version stamp in `final_provenance.json` changes). A rebuild of the figures gives the same pictures; the PDF files differ only in their embedded creation dates.

To re-run the experiments themselves (hours of GPU time; needs the photos, the encoder weights and a clean commit), use these entry points (each module's docstring lists its actions and settings files):

| Stage | Entry point |
|---|---|
| 0–7 | `uv run --locked python -m src.preflight --config configs/pipeline.yaml`, then `uv run --locked python -m src.pipeline --config configs/pipeline.yaml --stage <4\|5\|6\|7\|evaluate\|report\|all>` |
| 8 | `uv run python -m src.session_ablations fit`, `evaluate`, `report` |
| 9 | `uv run python -m src.partition_order <action>` |
| 10 | `uv run python -m src.paddy_zero_shot predict-heads`, `predict-cnn`, `report` |
| 11 | `uv run python -m src.committee select`, `plan`, `evaluate` |
| 12 | `uv run python -m src.explain select`, `heatmaps`, `plan` |
| 13 | `uv run python -m src.cost sizes`, `time --device gpu`, `time --device cpu` |

Run records name the commit that produced them by its id before the public release (the history was rewritten then to remove internal notes); `docs/commit_map.csv` gives the matching commit here, so `git checkout <new_commit>` shows the code of a run. That code is the same except for the rewording: the AI model raters carry neutral ids (`model_rater_a`, `model_rater_b`, `model_rater_c`), and comments no longer point to the removed notes.

Test-set files are evaluated once per frozen configuration; a `test_started.json` in a run folder marks that it was done and must not be deleted.

## 3. What is not included, and why

| Not included | Why |
|---|---|
| The photos (primary dataset and Paddy Doctor) | No images are redistributed. The only photo images in the paper are two figures (`final_samples`, `final_heatmaps`); the per-photo heat-map and blind-rating PNGs are not tracked |
| Encoder weights (DINOv3, DINOv2, ConvNeXt and the others) | Licence terms. They load through `pinned_pretrained()` at the pinned revisions listed in `data/DATASETS.md`; the DINOv3 licence is acknowledged in the paper |
| Tile-description caches (`cache/`) and model checkpoints | Size (about 104 GB of caches, 6.2 GB for the main tile cache alone). Rebuild with the Stage 6 pipeline; each cache passes the spec checks of `scripts/import_kaggle_cache.py` |
| The private Paddy Doctor copy on Kaggle (`aztrek/paddy-doctor-full`) | It is a private copy kept so that the photos are not redistributed; the original is at IEEE Dataport, DOI 10.21227/hz4v-af08 |
| Heat-map ratings by people | None exist: both sets of ratings are disclosed AI model ratings (`data/heatmap_ratings_provenance.txt`, `docs/reports/heatmap_ratings_report.md`) |

## 4. The paper and the notebook

`paper/ricetein_paper.pdf` is the paper, built from the sources in `paper/`. `notebooks/ricetein.ipynb` is the project notebook, built and executed with the command in section 2 and committed with its outputs (`.gitattributes` exempts this one notebook from nbstripout).
