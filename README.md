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
| Final paper | Broader experiments and Discussion/Conclusion deferred to the November 22 final |
| Follow-up study (audit of 2026-11-14) | Run and reported (`docs/reports/followup_report.md`): session-blocked sensitivity split, time-neighbour and strict-background controls, extended searches, same-head whole-photo control, block-bootstrap ranges and stability seeds. The nine core configurations lose 6.2–11.0 points of test macro-F1 on the session-blocked split. The session-blocked split is the paper's headline split (decided 2026-11-16; the locked split is reported beside it); every method was then tuned on its own validation photos and tested once (2026-11-17, `docs/reports/session_report.md`): attention MIL 77.9 ± 3.1% test macro-F1 on it, Swin-T 76.3 ± 4.1%, ConvNeXt-T 75.0 ± 5.0% |
| Stage 12 (explaining the model) | Code and tests ready on branch `stage12/explain` (`src/explain.py`, 2026-11-20; declared in handbook §24); the real runs (heat maps, deletion test, background score, report) wait for the GPU and the merge. The 50-heat-map human check is not started |
| Stage 6 (extra tile caches) | **Done (2026-11-21).** Built on Kaggle and imported with the spec checks: E3 DINOv2, E4 DINOv3-S, E2 ConvNeXt-T, 448 px tiles; built on `halo` and re-validated: Otsu tiles, native-resolution tiles. Each has 3,355 photos × 8 versions (26,840 files). The Paddy Doctor descriptions (Stage 10) are built on a second machine, the PC (rule of 2026-11-21, handbook §24) |
| Stage 9 (partition-order experiment) | **Done (2026-11-21, commit `ef95788`; report `docs/reports/partition_order_report.md`, table `results/tables/partition_order_inflation.csv`).** 30 runs, each scored once. Pooled test macro-F1 against the correct session-blocked protocol A (EfficientNetV2-S 75.9%, tile model 78.8%): augment-then-split B +17.2 and +11.5 points, random photo split C +8.8 and +4.0, train-and-score-on-everything D +11.0 and +10.1; all ranges above 0. A general lesson, not a comparison with any paper. Framing awaits the supervisor check |
| Stage 10 (Paddy Doctor zero-shot) | **Done (2026-11-21; report `docs/reports/paddy_report.md`, tables `results/tables/paddy_*.csv`).** Descriptions built on the PC and imported with the spec checks; 130 round-models scored once each. Macro-F1 on 8,086 field photos, native zoom: attention 26.1 ± 4.2%, ConvNeXt-T 30.1 ± 7.3%, Swin-T 27.0 ± 5.8%, EfficientNetV2-S 27.0 ± 5.5%, InceptionV3 16.6 ± 6.6% (uniform guessing: about 25%). **H3 is not shown** by the declared rule: attention beats one of the four fine-tuned 224 px models (InceptionV3) and shows no clear difference against the other three. |
| Environment (local machine + Kaggle GPUs) | Done and verified |
| Experiments | Stages 0–4 and the scoped nine-method core study are complete (3,334 eligible primary photos). Best mean test macro-F1: Swin-T 83.7 ± 1.3%; attention MIL: 82.4 ± 2.0%. Five rounds and original tuning budgets retained. On the session-blocked split, tuned on it, attention MIL has the best mean test macro-F1 (77.9 ± 3.1%; `docs/reports/session_report.md`). Other settings were run on the session split on 2026-11-21 (`docs/reports/session_conditions_report.md`): 128 px is clearly worse than 224 px for all four models, Otsu does not help, and ConvNeXt-T at 448 px reaches 79.2 ± 3.8% (above its 224 px result of 75.0 ± 5.0%; not yet compared with attention in a paired test). Stage 8 ablations on the session split (2026-11-21, `docs/reports/session_ablation_report.md`): on test, the common leaf scale beats native resolution (+2.9 macro-F1, 4 of 5 rounds) and the main configuration beats Otsu-cleaned tiles (+5.2, 5 of 5 rounds); the other nine variants were scored on validation only. Partition-order and Paddy evaluation are still to run. See `manuals/phase4_submission.md` and `docs/logbook.md`. |

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

Setup and the everyday Kaggle loop are in [`docs/KAGGLE_WORKFLOW.md`](docs/KAGGLE_WORKFLOW.md); the current state and next steps are in [`manuals/HANDOFF.md`](manuals/HANDOFF.md). Step-by-step instructions to reproduce every table will be added here once the results exist.

---

## Acknowledgements

- **Datasets:** Rice Diseases Image Dataset by H. M. Do (Kaggle, MIT licence); Paddy Doctor by A. Petchiammal et al. (CC BY 4.0).
- **DINOv3** by Meta AI (Siméoni et al., 2025), used under the DINOv3 License.
- **Software:** PyTorch, timm, Hugging Face Transformers and scikit-learn.
