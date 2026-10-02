# Dataset Records & Specifications

This document records the provenance, download metadata, file counts, and license terms for all datasets used in this project, fulfilling written commitment P1-12 from the Phase 1 specification.

---

## 1. Primary In-Domain Dataset: Rice Diseases Image Dataset

* **Role**: All model training, tuning, feature caching, and 5-fold cross-validation.
* **Source URL**: [https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset](https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset)
* **Kaggle Dataset ID**: `193945` (`minhhuy2810/rice-diseases-image-dataset`)
* **Download Date**: 2026-09-26
* **Downloaded By**: Ahsan Baig (`aztrek`) via Kaggle CLI
* **Archive Size**: 12,874,852,891 bytes (~12.0 GB)
* **Kaggle ETag / Validator**: `6007c8b467dd3134854a05e2dbd8083c`
* **Version**: Kaggle listing last updated **2019-11-15** (the dataset has a single release; use this date as the version identifier)
* **Local Path**: `data/raw/rice_diseases/`
* **License Terms**: **MIT License** (as stated on Kaggle landing page and API metadata).

### Verified Folder Structure & Class Counts

The official download unpacks into two directories:
1. `LabelledRice/Labelled/` (the master annotated image collection used as our primary ground truth)
2. `RiceDiseaseDataset/` (a pre-split train/validation folder: 400 per class for training, 123 per class for validation). **Verified (2026-10-02, MD5 checksums): all 2,092 files are exact byte-for-byte copies of files in `Labelled/`, with identical class labels; it contains no additional images.** It is ignored, so no photo is counted twice and our own 5-fold protocol is used instead.

Class counts derived directly from `LabelledRice/Labelled/`:

| Class Name | Observed Image Count | Published Count (Do, 2019) | Status / Notes |
|---|---|---|---|
| **BrownSpot** | 523 | 523 | Matches published count |
| **Healthy** | 1,488 | 1,488 | Matches published count |
| **Hispa** | 565 | 563 | +2 additional images in download |
| **LeafBlast** | 779 | 779 | Matches published count |
| **Total** | **3,355** | **3,353** | 3,355 high-resolution photos |

* **Exact duplicates inside `Labelled/`**: none (MD5 check, 2026-10-02). The 2 extra Hispa images are therefore not byte-identical copies; near-duplicate checks follow in Stage 2b.
* **File format**: all 3,355 files are `.jpg`; none carries an EXIF rotation tag.

* **Native Resolution — verified, and NOT uniform** (2026-10-02, image headers of all 3,355 files). The "1881 × 1881" figure reported in the literature (Ali et al. 2026) does not hold:
  * 1,293 distinct sizes; 3,341 square and 14 nearly square (e.g. 1881 × 1880).
  * Shorter side: minimum **734 px**, median **1,918 px**, maximum **3,120 px**. Most common sizes: 3120 × 3120 (573 images), 1881 × 1881 (105), 2172 × 2172 (34).
  * Shorter side by band: <1000 px: 22 · 1000–1499: 599 · 1500–1999: 1,245 · 2000–2499: 787 · 2500–2999: 121 · ≥3000: 581.
  * **Sizes differ by class** (median shorter side: BrownSpot 2,080, Healthy 2,000, Hispa 1,946, LeafBlast 1,736). This is a possible shortcut; the photo-properties-only baseline (handbook Stage 4b) tests it.
  * With 224-px tiles (handbook Stage 6 rule), a photo gives 16–196 tiles; **330,618 tiles in total, about 98.5 per photo**.

---

## 2. Secondary Out-of-Distribution Dataset: Paddy Doctor (Full Release)

* **Role**: Strictly held-out zero-shot evaluation on the 4 shared disease classes across 3 zoom levels (native, midway, matched). Never used for training or hyperparameter tuning.
* **Original DOI / Source**: Petchiammal et al., *Paddy Doctor: a visual image dataset for automated paddy disease classification*, CODS-COMAD 2023. IEEE Dataport, DOI: [10.21227/hz4v-af08](https://dx.doi.org/10.21227/hz4v-af08).
* **Private Kaggle Dataset ID**: `aztrek/paddy-doctor-full` (private, strictly un-redistributed per Phase 1 commitment).
* **Download / Upload Date**: 2026-09-26 / 2026-09-27
* **Owner / Manager**: Ahsan Baig (`aztrek`)
* **Archive Size**: ~4.65 GB (16,225 images + metadata.csv)
* **Local Path**: `data/raw/paddy_doctor/`
* **License Terms**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
* **Native Resolution**: Verified (all 16,225 headers, 2026-10-02): 16,219 images are $1080 \times 1440$ (portrait) and **6 are $1440 \times 1080$ (landscape)**. No EXIF rotation tags. `metadata.csv` matches the files on disk exactly (no missing or extra files; per-class counts identical).

### Verified Folder Structure & Class Counts

The dataset comprises 16,225 high-resolution smartphone images across 13 classes, accompanied by `metadata.csv` (16,225 rows: `image_id`, `label`, `variety`, `age`):

| Class Directory | Image Count | Mapping to Primary Task | Evaluation Role |
|---|---|---|---|
| `normal` | 2,405 | **Healthy** | Zero-shot evaluation |
| `blast` | 2,351 | **LeafBlast** | Zero-shot evaluation |
| `hispa` | 2,151 | **Hispa** | Zero-shot evaluation (diagnostic) |
| `brown_spot` | 1,257 | **BrownSpot** | Zero-shot evaluation |
| `tungro` | 1,951 | *(unmapped)* | Zoom calibration / leaf measurement |
| `white_stem_borer` | 1,273 | *(unmapped)* | Zoom calibration / leaf measurement |
| `leaf_roller` | 1,095 | *(unmapped)* | Zoom calibration / leaf measurement |
| `downy_mildew` | 868 | *(unmapped)* | Zoom calibration / leaf measurement |
| `yellow_stem_borer` | 765 | *(unmapped)* | Zoom calibration / leaf measurement |
| `bacterial_leaf_blight` | 648 | *(unmapped)* | Zoom calibration / leaf measurement |
| `black_stem_borer` | 506 | *(unmapped)* | Zoom calibration / leaf measurement |
| `bacterial_leaf_streak` | 505 | *(unmapped)* | Zoom calibration / leaf measurement |
| `bacterial_panicle_blight`| 450 | *(unmapped)* | Zoom calibration / leaf measurement |
| **Total** | **16,225** | **8,164 shared-class images** | **Full 13-class uncompressed release** |

* **Status**: Complete and verified (counts, metadata and image sizes).
* **Still to add**: SHA-256 checksum of the downloaded IEEE DataPort archive, if the original zip is still available.


---

## 3. Pretrained Model Weights: DINOv3 (private Kaggle copy)

* **Role**: Frozen encoders E1 (ViT-B/16) and E4 (ViT-S/16); E5 (ViT-L/16) is optional and not uploaded yet.
* **Source**: Hugging Face (gated; access granted to the team), web-image versions `lvd1689m`.
* **Private Kaggle copy**: `aztrek/ricetein-weights`, uploaded 2026-10-02 by Ahsan Baig with `scripts/kaggle_upload_weights.sh`. Includes `LICENSE.md` and `REVISIONS.txt`. Kaggle notebooks load from it via `pretrained_source()`, so every run uses these exact weights.
* **Pinned versions (Hugging Face commits)**:

| Model | Commit |
|---|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | `114c1379950215c8b35dfcd4e90a5c251dde0d32` |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | `5931719e67bbdb9737e363e781fb0c67687896bc` |

* **License Terms**: **DINOv3 License** (Meta, last updated 19 August 2025). Copies and redistribution are allowed only with the licence attached (§1.b.i); publications must acknowledge the use of DINOv3 (§1.b.ii). The Kaggle copy is private.
* **Laptop note**: the laptop loads the same models from Hugging Face. To pin the laptop too, pass the commit above as `revision=` when loading.

---

## 4. Optional Datasets (Future Work / Ablations)

* **Dhan-Shomadhan** (Mendeley, DOI: 10.17632/znsxdctwtt.1): Secondary zero-shot candidate if time permits.
* **RiceSeg-5932** (Mendeley, 2025): Pixel-level segmentation masks for quantitative lesion attention overlap testing.
