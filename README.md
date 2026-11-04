# ricetein: Resolution-Preserving Weakly Supervised Recognition of Rice Leaf Diseases

Rice leaf photos are usually shrunk to a small thumbnail (128 or 224 pixels) before a model sees them, which throws away most of the fine detail that tells one disease from another. This project keeps that detail:

1. Each high-resolution photo is cut into small **full-detail tiles** (224 × 224 pixels, 16–196 per photo).
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
| Phase 4: Experiments and Results | In progress |
| Final paper | Not started |
| Environment (local machine + Kaggle GPUs) | Done and verified |
| Experiments | Stage 0 done (2026-11-04); Stage 1 next (see `docs/HANDBOOK.md`) |

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
  WORKFLOW_GUIDE.md   day-to-day: running notebooks on the local machine and on Kaggle
  logbook.md        dated log of what was done, found and decided
src/             project code (installed as a package, so `import src` works anywhere)
scripts/         Kaggle helper scripts (sync code, push, pull, upload weights)
kaggle/          Kaggle settings for each dataset and notebook
data/            DATASETS.md (raw data in data/raw/, not in git)
runs/            one folder per experiment run (settings, scores, predictions, log)
deliverables/    submitted documents (PDFs) with the LaTeX sources they were built from
paper/           working LaTeX sources of the article (shared ref.bib, one file per chapter)
manuals/         per-stage to-do guides
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

## Running experiments

- **On the local machine:** open a notebook from `notebooks/` with the project's `.venv` kernel. Every run writes its record to `runs/<run_name>/`.
- **On Kaggle GPUs (heavy jobs), from the terminal:**
  ```bash
  ./scripts/kaggle_sync_code.sh          # upload the committed code to Kaggle
  ./scripts/kaggle_push.sh <notebook>    # start a background run on a T4 GPU
  ./scripts/kaggle_pull.sh <notebook>    # fetch results into runs/ when it's finished
  ```
  Kaggle notebooks are thin launchers that run the project code in an isolated environment with the exact locked package versions.

Setup details are in [`docs/KAGGLE_WORKFLOW.md`](docs/KAGGLE_WORKFLOW.md), daily use in [`docs/WORKFLOW_GUIDE.md`](docs/WORKFLOW_GUIDE.md). Step-by-step instructions to reproduce every table will be added here once the results exist.

---

## Acknowledgements

- **Datasets:** Rice Diseases Image Dataset by H. M. Do (Kaggle, MIT licence); Paddy Doctor by A. Petchiammal et al. (CC BY 4.0).
- **DINOv3** by Meta AI (Siméoni et al., 2025), used under the DINOv3 License.
- **Software:** PyTorch, timm, Hugging Face Transformers and scikit-learn.
