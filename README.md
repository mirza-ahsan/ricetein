# ricetein: Resolution-Preserving Weakly Supervised Recognition of Rice Leaf Diseases

Rice leaf photos are usually shrunk to a small thumbnail (128 or 224 pixels) before a model sees them, which throws away most of the fine detail that tells one disease from another. This project keeps that detail:

1. Each high-resolution photo is cut into small **high-detail tiles** (224 × 224 pixels; each photo is first brought to a common leaf scale, giving 49–100 tiles).
2. Every tile is described once by a **frozen, self-supervised image model** (Meta's DINOv3).
3. A small **attention model** (about 0.26 M trainable parameters), trained only from whole-photo labels, learns which tiles show the disease.

The attention weights also produce a **heat map** of where the model looked.

The work is evaluated with a strict, leakage-free protocol:
- a **5-fold split** fixed before any augmentation, keeping near-duplicate photos together;
- **macro-F1 and balanced accuracy**, with spread across folds;
- **zero-shot testing** on an independent field dataset;
- checks that the heat maps reflect what the model really uses;
- a measurement of compute cost.

It is compared against **InceptionV3, EfficientNetV2-S, ConvNeXt-T and Swin-T**, fine-tuned on whole photos at several resolutions, with and without Otsu background removal.

Course project at FAST School of Computing, FAST-NUCES Lahore, and the basis of a research paper.

**Team:** Ahsan Baig (23L-0515), Harris Tabassum (23L-0551), Muhammad Umer (23L-0845)

---

## Status

| Part | State |
|---|---|
| Phase 1: Dataset Description | Submitted (`deliverables/phase1/`) |
| Phase 2: Introduction and Literature Review | Submitted (`deliverables/phase2/`) |
| Phase 3: Methodology and Experimental Setup | Final (`deliverables/phase3/`) |
| Phase 4: Experiments and Results | Core-study PDF and notebook ready locally in `submission/phase4/`; not uploaded to the course portal |
| **Development** | **Complete (2026-11-21, tag `dev-complete-20261121`).** Every experiment is done and every final table and figure is rebuilt from `runs/` (`results/tables/final_*`, `results/figures/final_*`). Only `paper/`, the documents and the packaging tooling change from here on; no experiment is re-run. Dhan-Shomadhan is future work. |
| Final paper | **Stage 14 draft written (2026-11-22)**: `paper/paper.tex` (IEEE conference, about 10 pages of main text, 2-page appendix; build: `paper/BUILD.md`). The pipeline diagram is drawn in TikZ. Next: the manager's review edits, then packaging (the submission notebook). Deadline 2026-11-22 23:59 |
| Follow-up study (audit of 2026-11-14) | Run and reported (`docs/reports/followup_report.md`): session-blocked sensitivity split, time-neighbour and strict-background controls, extended searches, same-head whole-photo control, block-bootstrap ranges and stability seeds. The nine core configurations lose 6.2–11.0 points of test macro-F1 on the session-blocked split. The session-blocked split is the paper's headline split (decided 2026-11-16; the locked split is reported beside it); every method was then tuned on its own validation photos and tested once (2026-11-17, `docs/reports/session_report.md`): attention MIL 77.9 ± 3.1% test macro-F1 on it, Swin-T 76.3 ± 4.1%, ConvNeXt-T 75.0 ± 5.0% |
| Stage 12 (explaining the model) | **Done (2026-11-21).** Deletion test, background score, heat maps and 50 blind heat maps rated by three disclosed model raters (`docs/reports/explain_report.md`). Test photos: removing the most-attended tiles costs far more than removing random tiles (gap 25.4 points at k=16; verdict under both ranges: top-k matters more than random-k). Paddy photos (photo range only): gap 2.2 points at k=16, attention on background no different from the tile share. Ratings: yes 39.3%, partly 32.0%, no 28.7%, all agree on 88.0%. A second set of three agent ratings (model raters, set 2; the importer's kind label is `human`, but no person rated the maps; `docs/reports/heatmap_ratings_report.md`): yes 49.3%, partly 24.0%, no 26.7%, all agree on 76.0% |
| Stage 6 (extra tile caches) | **Done (2026-11-21).** Built on Kaggle and imported with the spec checks: E3 DINOv2, E4 DINOv3-S, E2 ConvNeXt-T, 448 px tiles; built on `halo` and re-validated: Otsu tiles, native-resolution tiles. Each has 3,355 photos × 8 versions (26,840 files). The Paddy Doctor descriptions (Stage 10) are built on a second machine, the PC (rule of 2026-11-21, handbook §24) |
| Stage 9 (partition-order experiment) | **Done (2026-11-21, commit `ef95788`; report `docs/reports/partition_order_report.md`, table `results/tables/partition_order_inflation.csv`).** 30 runs, each scored once. Pooled test macro-F1 against the correct session-blocked protocol A (EfficientNetV2-S 75.9%, tile model 78.8%): augment-then-split B +17.2 and +11.5 points, random photo split C +8.8 and +4.0, train-and-score-on-everything D +11.0 and +10.1; all ranges above 0. A general lesson, not a comparison with any paper. Framing is checked by Ahsan before anything is submitted |
| Stage 10 (Paddy Doctor zero-shot) | **Done (2026-11-21; report `docs/reports/paddy_report.md`, tables `results/tables/paddy_*.csv`).** Descriptions built on the PC and imported with the spec checks; 130 round-models scored once each. Macro-F1 on 8,086 field photos, native zoom: attention 26.1 ± 4.2%, ConvNeXt-T 30.1 ± 7.3%, Swin-T 27.0 ± 5.8%, EfficientNetV2-S 27.0 ± 5.5%, InceptionV3 16.6 ± 6.6% (uniform guessing: about 25%). **H3 is not shown** by the declared rule: attention beats one of the four fine-tuned 224 px models (InceptionV3) and shows no clear difference against the other three. |
| Stage 13 (cost and compression) | **Done (2026-11-21; report `docs/reports/cost_report.md`, tables `results/tables/cost_results.csv`, `cost_compression.csv`).** Measured on `halo` (RTX 4060, Ryzen 7 7700; hardware-dependent). Attention on E1: 0.31 s per photo on the GPU, 5.4 s on the CPU, 328 MB; whole-image models 0.05 s and 0.06 to 0.12 s, 78 to 106 MB: on the CPU the tile method is about 80 times slower than a 224 px model and 47 times slower than ConvNeXt-T at 448 px. Validation macro-F1 (reference 78.5): 16 tiles chosen by a colour rule 76.5 at 1.2 s per photo on the CPU; E4 74.8; distilled E4 78.6; 8-bit encoder 23.6 on a 399-photo subset (the 8-bit encoder fails). Final small model (16 tiles), tested once: 76.7 against attention 77.9, no clear difference (+1.4, 3 of 5 rounds); ONNX on the CPU 0.93 s per photo |
| Stage 11 (committee) | **Done (2026-11-21).** Committees chosen on validation only and tested once on the session test photos (`runs/committee_selection.json`, `runs/committee_plan.json`, `runs/committee_*/`). Test macro-F1: attention 77.9 ± 3.1; committee with tile models (ConvNeXt-T 448 px + E1 attention) 81.6 ± 3.6, beats attention under both ranges (+3.6, 5 of 5 rounds); committee of whole-image models (ConvNeXt-T 448 px + Swin-T) 80.3 ± 4.0, beats attention on macro-F1 under both ranges and on balanced accuracy under the photo range only. On Paddy Doctor (native zoom) attention beats the tile committee (26.1 against 23.0). Eight-view TTA of ConvNeXt-T 448 px: 79.2 to 81.3. Report `docs/reports/committee_report.md`. **Paired test against ConvNeXt-T 448 px alone (declared post-hoc 2026-11-21, `docs/reports/paired_committee_report.md`):** the committee beats it under both ranges on macro-F1 (+2.2, photo range [+1.2, +3.2], block range [+1.0, +3.8]) and on balanced accuracy (+2.3), 4 of 5 rounds each; ConvNeXt-T is a member of the committee, so the two are not independent |
| Environment (local machine + Kaggle GPUs) | Done and verified |
| Experiments | Stages 0–4 and the scoped nine-method core study are complete (3,334 eligible primary photos). Best mean test macro-F1: Swin-T 83.7 ± 1.3%; attention MIL: 82.4 ± 2.0%. Five rounds and original tuning budgets retained. On the session-blocked split, tuned on it, attention MIL reaches 77.9 ± 3.1% test macro-F1 (`docs/reports/session_report.md`). Other settings were run on the session split on 2026-11-21 (`docs/reports/session_conditions_report.md`): 128 px is clearly worse than 224 px for all four models, Otsu does not help, and ConvNeXt-T at 448 px reaches 79.2 ± 3.8% (above its 224 px result of 75.0 ± 5.0%). A declared post-hoc paired comparison (2026-11-21, `docs/reports/paired_conditions_report.md`) finds no clear difference between attention and ConvNeXt-T at 448 px (attention minus ConvNeXt-T: −1.5 macro-F1, photo range [−3.0, 0.2], block range [−3.3, 0.9], attention wins 1 of 5 rounds), and attention beats the other eight conditions under both ranges. Stage 8 ablations on the session split (2026-11-21, `docs/reports/session_ablation_report.md`): on test, the common leaf scale beats native resolution (+2.9 macro-F1, 4 of 5 rounds) and the main configuration beats Otsu-cleaned tiles (+5.2, 5 of 5 rounds); the other nine variants were scored on validation only. Partition-order and Paddy evaluation are done (Stages 9 and 10, rows above). See `manuals/phase4_submission.md` and `docs/logbook.md`. |

---

## Datasets

| Dataset | Role | Size | Licence |
|---|---|---|---|
| [Rice Diseases Image Dataset](https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset) (Do, 2019) | Training, tuning, in-domain testing | 3,355 photos, 4 classes, 734–3,120 px | MIT |
| [Paddy Doctor](https://dx.doi.org/10.21227/hz4v-af08) (Petchiammal et al., 2023), full release | Zero-shot testing only (never trained on) | 16,225 field photos, 13 classes; 8,164 in the 4 shared classes | CC BY 4.0 |

The 4 classes are Healthy, Leaf blast, Brown spot and Hispa. **No images are stored in or redistributed by this repository**; downloads go to `data/raw/`, which git ignores. Provenance, verified counts and image sizes are in [`data/DATASETS.md`](data/DATASETS.md).

---

## Repository layout

```
docs/            project documents (start here)
  HANDBOOK.md       the full practical plan: stages 0–14, rules, decisions, file build order
  DELIVERABLES.md   coursework tracker: deliverables, marks, written commitments
  KAGGLE_WORKFLOW.md  one-time setup for running notebooks on Kaggle GPUs
  logbook.md        dated log of what was done, found and decided
src/             project code (installed as a package, so `import src` works anywhere)
scripts/         Kaggle helper scripts (sync code, push, pull, upload weights)
kaggle/          Kaggle settings for each dataset and notebook
data/            DATASETS.md (raw data in data/raw/, not in git)
runs/            one folder per experiment run (settings, scores, predictions, log)
deliverables/    submitted documents (PDFs) with the LaTeX sources they were built from
paper/           working LaTeX sources of the article (shared ref.bib, one file per chapter)
manuals/         HANDOFF.md: where the project stands and what to do next (start here)
```

More folders (`configs/`, `splits/`, `notebooks/`, `tests/`, `results/`) are added as the stages need them. The full planned tree is in `docs/HANDBOOK.md` §9.2.

---

## Setup

Requirements: Linux, an NVIDIA GPU with a CUDA 12.8-capable driver (development uses an RTX 4060, 8 GB), [uv](https://docs.astral.sh/uv/), git.

**1. Clone and install** (Python 3.12.13, pinned in `.python-version` to match Kaggle; PyTorch 2.11 with CUDA 12.8; every package version pinned in `uv.lock`):
```bash
git clone git@github.com:mirza-ahsan/ricetein.git
cd ricetein
uv sync
uv run nbstripout --install     # strips notebook outputs on commit (needed once per clone)
```

**2. Kaggle access.** Get a token from kaggle.com → Settings → API → Create New Token, then:
```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
uv run kaggle datasets list -s "rice diseases"     # test
```

**3. Data** (about 18 GB):
```bash
uv run kaggle datasets download minhhuy2810/rice-diseases-image-dataset -p data/raw/rice_diseases --unzip
uv run kaggle datasets download aztrek/paddy-doctor-full -p data/raw/paddy_doctor --unzip   # private; team members only
```
Outside the team, download the full Paddy Doctor release from IEEE DataPort instead.

**4. DINOv3** is gated on Hugging Face. Request access on the [model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), then log in with a read token:
```bash
uv run hf auth login
```

**5. Check everything works** (packages, GPU, both datasets, DINOv3):
```bash
uv run python -m src.smoke_check --require-dinov3 --run-name smoke_mine
```

---

## Phase 4 submission

Today's agreed scope is the core study, with original training/search budgets retained. See [the submission guide](manuals/phase4_submission.md). Working paper: `paper/phase4_results.tex`; code-containing notebook: `notebooks/phase4_submission.ipynb`. Draft paper builds permit explicit missing-evidence messages; final builds require completed evidence. The broader Stage 8–13 programme is reserved for the final paper.

## Running experiments

- **On the local machine:** open a notebook from `notebooks/` with the project's `.venv` kernel. Every run writes its record to `runs/<run_name>/`.
- **On Kaggle GPUs (heavy jobs), from the terminal:**
  ```bash
  ./scripts/kaggle_sync_code.sh          # upload the committed code to Kaggle
  ./scripts/kaggle_push.sh <notebook>    # start a background run on a T4 GPU
  ./scripts/kaggle_pull.sh <notebook>    # fetch results into runs/ when it's finished
  ```
  Kaggle notebooks are thin launchers that run the project code in an isolated environment with the exact locked package versions.

Setup and the everyday Kaggle loop are in [`docs/KAGGLE_WORKFLOW.md`](docs/KAGGLE_WORKFLOW.md); the current state and next steps are in [`manuals/HANDOFF.md`](manuals/HANDOFF.md). How to rebuild every table and figure is in the next section.

## How to reproduce each table

Every final table and figure is generated from the run records in `runs/` and the per-stage tables that were themselves generated from them; nothing is typed by hand. A result whose stage is not finished is written as *pending: <stage>*.

| Output | Command | Reads |
|---|---|---|
| All final tables, `results/tables/final_*.csv` and `final_*.tex` (main, conditions, per-class recall with hispa, confusion, paired tests, ablations, partition order, shortcut controls, committee, deletion test, cost) and `final_provenance.json` (source files with SHA-256) | `uv run python -m src.make_tables` | `runs/`, `results/tables/` |
| All final figures, `results/figures/final_*.{png,pdf}` and `final_figures.json` (sample photos need `data/raw/`, so they are built only in a checkout that has the photos) | `uv run python -m src.make_figures` | `runs/`, `results/tables/`, `results/explain/` |
| What is present and what is pending (tables, figures, dataset reports, every written commitment of `docs/DELIVERABLES.md` §6), `final_readiness.json` | `uv run python -m src.make_tables --check` | the two outputs above |
| The final submission notebook `submission/final/ricetein_final.ipynb` (git-ignored; heavy steps are shown, not re-run) | `uv run python -m src.build_submission_notebook` | `src/`, `notebooks/`, `results/`, `docs/reports/` |
| The per-stage reports and tables the final tables read | `python -m src.followup_report`, `session_report`, `session_conditions_report`, `session_ablations report`, `partition_report`, `paddy_zero_shot report`, `explain report` | `runs/` |

`notebooks/14_results.ipynb` runs the first three in order.

---

## Acknowledgements

- **Datasets:** Rice Diseases Image Dataset by H. M. Do (Kaggle, MIT licence); Paddy Doctor by A. Petchiammal et al. (CC BY 4.0).
- **DINOv3** by Meta AI (Siméoni et al., 2025), used under the DINOv3 License.
- **Software:** PyTorch, timm, Hugging Face Transformers and scikit-learn.
