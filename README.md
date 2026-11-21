# Resolution-Preserving Weakly Supervised Recognition of Rice Leaf Diseases

**Ahsan Baig, Harris Tabassum and Muhammad Umer**
FAST School of Computing, National University of Computer and Emerging Sciences (FAST-NUCES), Lahore, Pakistan

Paper: [`paper/ricetein_paper.pdf`](paper/ricetein_paper.pdf) · Notebook: [`notebooks/ricetein.ipynb`](notebooks/ricetein.ipynb)

Rice leaf photos are taken at high resolution, but most recognition systems shrink them to a few hundred pixels, which throws away lesion-scale detail. We study a resolution-preserving alternative that needs only photo-level labels:

1. **Tile at a common leaf scale.** Each photo (734–3,120 px) is rescaled so that leaves have about the same width in pixels, then cut into 224 px tiles (49–100 per photo, 265,349 in all).
2. **Describe once.** A frozen **DINOv3 ViT-B/16** encoder describes every tile; the descriptions are cached.
3. **Decide.** A small **gated-attention multiple-instance learning (MIL) head** (Ilse et al., 2018; about 0.23 M parameters), trained from the photo label alone, combines the tile descriptions. Its attention weights give a heat map.

Four classes: healthy, leaf blast, brown spot and hispa. The method is compared with InceptionV3, EfficientNetV2-S, ConvNeXt-T and Swin-T fine-tuned on whole photos (128, 224 and 448 px, with and without Otsu background removal), linear probes, mean and max pooling, and ablations.

## Key results

All numbers are from the final tables in `results/tables/`, which are built from the run records in `runs/`. Scores are percentages, mean ± SD over 5 rounds.

**Session-blocked split, test photos** (`final_main.csv`, `final_committee.csv`):

| Method | Test macro-F1 | Test balanced accuracy |
|---|---|---|
| InceptionV3 (224 px) | 72.0 ± 3.6 | 72.2 ± 4.9 |
| EfficientNetV2-S (224 px) | 74.1 ± 4.8 | 73.9 ± 5.9 |
| ConvNeXt-T (224 px) | 75.0 ± 5.0 | 75.8 ± 3.7 |
| Swin-T (224 px) | 76.3 ± 4.1 | 75.8 ± 4.8 |
| **Attention MIL (ours)** | **77.9 ± 3.1** | **78.0 ± 3.6** |
| ConvNeXt-T (448 px) | 79.2 ± 3.8 | 79.1 ± 5.0 |
| Committee: ConvNeXt-T 448 px + attention MIL | 81.6 ± 3.6 | 81.4 ± 4.5 |

- By the paired-bootstrap rule below, attention MIL is clearly above two of the four whole-image models at 224 px (InceptionV3 and EfficientNetV2-S), and above both whole-photo controls that use the same encoder and head (`final_paired_session.csv`).
- It is **not clearly different** from ConvNeXt-T at 448 px, which has the higher mean: the difference is −1.5 points, with 95% ranges of [−3.0, 0.2] by photo and [−3.3, 0.9] by capture block (`final_paired_conditions.csv`).
- The committee of the two scores higher than either. Its comparison with ConvNeXt-T at 448 px, one of its own members, was declared after the means were seen: +2.2 points, 4 of 5 rounds (`final_paired_committee.csv`).
- On test, tiles at the common leaf scale beat tiles at native resolution (+2.9 points) and Otsu-cleaned tiles (+5.2 points) (`final_ablations_test.csv`).

**Zero-shot transfer to field photos is not shown** (`final_main.csv`). On 8,086 Paddy Doctor photos of the same four classes, with nothing trained or tuned on them, every method scores low. Attention MIL reaches 26.1 ± 4.2 macro-F1 at native zoom, and the four whole-image models at 224 px reach 16.6 to 30.1. The pre-declared hypothesis (attention MIL beats each of these four) is not shown: it beats only InceptionV3.

**Deletion test** (`final_explain.csv`, test photos): removing the 16 most-attended tiles lowers macro-F1 from 77.9 to 15.6, while removing 16 random tiles lowers it to 75.3. The true-class probability falls 25.4 points more for the attended tiles (5 of 5 rounds). On Paddy Doctor photos the effect is small.

**Partition order** (`final_partition_order.csv`): augmenting the photos before splitting them raises pooled test macro-F1 by 17.2 points (EfficientNetV2-S) and 11.5 points (attention MIL) over the correct protocol. Split order matters.

**Cost** (`final_cost.csv`; one machine, RTX 4060 and Ryzen 7 7700): the tile method needs about 79 encoder passes per photo. Attention MIL takes 0.31 s per photo on the GPU and 5.36 s on the CPU (328 MB on disk); ConvNeXt-T at 448 px takes 0.0472 s and 0.115 s (106 MB).

## Testing protocol

- **Near-duplicates are grouped** (perceptual hashes, encoder similarity and a checked review) and never cross folds.
- **Session-blocked 5-fold split** (`splits/session_blocked/folds.csv`): photos from one capture session stay in one fold. The split was made once, before any augmentation, and committed. A locked split that groups near-duplicates only is reported beside it.
- **Every method gets the same tuning budget** and is selected on validation macro-F1 averaged over the 5 rounds. **Test photos are evaluated once per frozen configuration.** Augmented or cached copies never reach validation or test; the code enforces this and `tests/test_leakage.py` and `tests/test_mil_leakage.py` check it.
- **"A beats B"** only if the 95% paired-bootstrap range of the difference is above 0 (resampled by photo, and by capture block) **and** A wins at least 4 of the 5 rounds. Otherwise the result is "no clear difference". Negative results are reported.
- **Paddy Doctor is never used for training, tuning or selection.** Its zero-shot plan was written and frozen before the run (`docs/reports/paddy_pre_declaration.md`).
- Some judgements were made by disclosed AI model raters: the background audit, the duplicate review, the label check, the Otsu quality check, one of the two leaf-width measurers, and both sets of heat-map ratings (`model_rater_*`; the second heat-map set is stored under the importer's label `human_rater_1..3`, see `data/heatmap_ratings_provenance.txt`).

## Repository layout

| Folder | What it holds |
|---|---|
| `src/` | All project code (installed as a package, so `import src` works anywhere) |
| `configs/` | Settings files for every stage (`base.yaml` holds the shared settings and the pinned model revisions) |
| `splits/` | The committed split files: locked, session-blocked, Paddy Doctor, and the deliberately faulty partition-order splits |
| `data/` | Dataset records (`DATASETS.md`), the per-photo inventory, and the audit and review files. No photos |
| `runs/` | One folder per run: config, commit, split checksum, data version, metrics and per-photo predictions (no checkpoints) |
| `results/` | Final tables and figures built from `runs/`, with source checksums; deletion-test files |
| `docs/` | `REPRODUCIBILITY.md`, `commit_map.csv`, and one report per stage in `reports/` |
| `paper/` | The paper PDF and its LaTeX sources (`paper.tex`, `sections/`, `tables/`, `fig/`, `ref.bib`, `BUILD.md`) |
| `notebooks/` | Thin launcher notebooks for each stage, and the executed project notebook `ricetein.ipynb` |
| `scripts/`, `kaggle/` | Helper scripts, and the settings for the Kaggle GPU runs |
| `tests/` | The test suite, including the leakage checks |

## Setup

Requirements: Linux, an NVIDIA GPU with a CUDA 12.8-capable driver, [uv](https://docs.astral.sh/uv/) and git. The experiments ran on an RTX 4060 (8 GB), an RTX 3070 (8 GB) and Kaggle T4 GPUs.

```bash
git clone https://github.com/mirza-ahsan/ricetein.git
cd ricetein
uv sync                      # Python 3.12.13, PyTorch 2.11 (CUDA 12.8); every version pinned in uv.lock
uv run nbstripout --install  # strips notebook outputs on commit
uv run --locked pytest -q    # the test suite needs no data and no GPU
```

DINOv3 is gated on Hugging Face: request access on the [model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), then run `uv run hf auth login` with a read token. The encoders load at the revisions pinned in `configs/base.yaml`.

## Data

**No images are included in this repository or in its history.** Download both datasets yourself into `data/raw/`, which git ignores:

| Dataset | Role | Where | Licence |
|---|---|---|---|
| Rice Diseases Image Dataset (Do, 2019): 3,355 photos, 4 classes | Training, tuning and in-domain testing | [Kaggle](https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset): `uv run kaggle datasets download minhhuy2810/rice-diseases-image-dataset -p data/raw/rice_diseases --unzip` | MIT |
| Paddy Doctor (Petchiammal et al., 2023), full release | Zero-shot testing only | [IEEE DataPort, DOI 10.21227/hz4v-af08](https://dx.doi.org/10.21227/hz4v-af08), into `data/raw/paddy_doctor/` | CC BY 4.0 |

Versions, download dates, archive checksums and verified counts are in [`data/DATASETS.md`](data/DATASETS.md). Then check the environment with `uv run python -m src.smoke_check --require-dinov3 --run-name smoke_mine`.

## Reproducing the results

Every table and figure is generated from the run records in `runs/`; nothing is typed by hand. From the repository root:

| What | Command |
|---|---|
| Final tables (`results/tables/final_*`) | `uv run --locked python -m src.make_tables` |
| Final figures (`results/figures/final_*`; the sample-photo figures need `data/raw/`) | `uv run --locked python -m src.make_figures` |
| Check that every promised table, figure and commitment is present | `uv run --locked python -m src.make_tables --check` |
| Copy the tables and figures into the paper | `./scripts/sync_paper_assets.sh` |
| The paper | see `paper/BUILD.md` (pdfLaTeX and biber; `paper.tex` is the main file) |
| The notebook (`notebooks/ricetein.ipynb`) | `uv run --locked python -m src.build_notebook`, then execute it with `jupyter nbconvert --execute` |

Re-running the experiments themselves takes many GPU hours. It needs the photos, the encoder weights and the tile-description caches (about 104 GB, not included; the Stage 6 pipeline rebuilds them). The entry points of every stage, and what is not included and why, are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Each run record names the commit that produced it. The history of this repository was rewritten before its public release, so the records carry the original commit ids; [`docs/commit_map.csv`](docs/commit_map.csv) maps each one to its commit here.

## Citation

```bibtex
@misc{baig2026ricetein,
  title  = {Resolution-Preserving Weakly Supervised Recognition of Rice Leaf Diseases},
  author = {Baig, Ahsan and Tabassum, Harris and Umer, Muhammad},
  year   = {2026},
  note   = {FAST-NUCES Lahore. Code and run records: https://github.com/mirza-ahsan/ricetein}
}
```

## Licence and acknowledgements

The code, run records and documents in this repository are released under the [MIT licence](LICENSE). The datasets and the model weights keep their own licences and are **not redistributed** here:

- **DINOv3** (Siméoni et al., 2025) is used under the DINOv3 License; as that licence requires, we acknowledge its use. Versions: `facebook/dinov3-vitb16-pretrain-lvd1689m` and `facebook/dinov3-vits16-pretrain-lvd1689m`, at the revisions pinned in `configs/base.yaml`.
- **Rice Diseases Image Dataset** by H. M. Do (MIT licence); **Paddy Doctor** by Petchiammal et al. (CC BY 4.0).
- Software: PyTorch, timm, Hugging Face Transformers and scikit-learn.
