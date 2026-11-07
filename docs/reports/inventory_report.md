# Stage 1 inventory report

Built by `src/inventory_report.py` from `data/inventory.csv` and `data/paddy_inventory.csv`. Every number is computed; the only typed inputs are the published counts and the record of which hash groups were inspected by eye (`configs/inventory.yaml`, `report`).

## 1. Primary dataset: class counts

| class | folder | found | published (Do 2019) | difference |
|---|---|---|---|---|
| healthy | Healthy | 1488 | 1488 | 0 |
| leaf blast | LeafBlast | 779 | 779 | 0 |
| brown spot | BrownSpot | 523 | 523 | 0 |
| hispa | Hispa | 565 | 563 | 2 |

Total **3,355** files in `LabelledRice/Labelled/` (published: 3,353). The extra photos (hispa +2) are not exact copies of any other photo (section 3).

The download's other folder(s) (`RiceDiseaseDataset/`) hold 2,092 files: 2,092 are byte-identical copies of inventoried photos and 2,092 of those carry the same class. They add no photos, so they are ignored and no photo is counted twice.

## 2. File integrity

| dataset | files | decode fully | formats | modes | rotation tags |
|---|---|---|---|---|---|
| primary | 3,355 | 3,355 | JPEG | RGB | 0 |
| Paddy Doctor | 16,225 | 16,225 | JPEG | RGB | 0 |

Every file decodes fully; formats: JPEG; colour modes: RGB; no file carries a rotation tag, so photos are read as stored and orientation handling cannot differ between machines.

## 3. Exact duplicates (same SHA-256)

- **Primary:** 0 duplicate groups.
- **Between the datasets:** 0 shared files.
- **Paddy Doctor:** 144 groups covering 301 files (groups of 2: 135, groups of 3: 6, groups of 4: 2, groups of 5: 1). In the 4 classes used for zero-shot testing: 152 files in 74 groups, i.e. 78 extra copies. 1 group(s) span more than one class (`downy_mildew/PDD06425.jpg`, `tungro/PDD12798.jpg`); none of them is in the 4 test classes. Every group has the same variety and age in `metadata.csv`, so these look like files stored twice in the release, not separate photos.

**To decide in Stage 3** (`splits/paddy4.csv`): count each image once.

## 4. Photo size by class (a possible shortcut)

Shorter side in pixels.

| class | photos | shortest | median | longest | at 3120 px | median MB |
|---|---|---|---|---|---|---|
| healthy | 1488 | 826 | 2000 | 3120 | 18% | 2.47 |
| leaf blast | 779 | 734 | 1736 | 3120 | 9% | 1.58 |
| brown spot | 523 | 812 | 2080 | 3120 | 23% | 2.18 |
| hispa | 565 | 877 | 1946 | 3120 | 21% | 2.39 |

Photos are square or nearly so (longer side at most 0.06% longer than the shorter); the shorter side runs from 734 to 3120 px (median 1918). The class with the smallest photos is **leaf blast** (median 1736 px, against 2080 px for brown spot). A model could pick up size instead of disease; the photo-properties baseline (Stage 4b) measures how much.

## 5. Camera, capture day and JPEG quality by class

**Camera (EXIF make and model).**

| class | Apple iPhone 6s Plus | Apple iPhone X | Xiaomi Redmi Note 4 |
|---|---|---|---|
| healthy | 4 (<1%) | 58 (4%) | 1,426 (96%) |
| leaf blast | 111 (14%) | 60 (8%) | 608 (78%) |
| brown spot | 18 (3%) | 48 (9%) | 457 (87%) |
| hispa | 0 (0%) | 0 (0%) | 565 (100%) |

3 cameras appear. Share of photos not taken with the most common one (Xiaomi Redmi Note 4): healthy 4%, leaf blast 22%, brown spot 13%, hispa 0%. A model could pick up the camera instead of the disease; Stage 4b adds a declared camera-only baseline next to the methodology's baselines.

**Capture day (EXIF DateTimeOriginal).**

| capture day | healthy | leaf blast | brown spot | hispa |
|---|---|---|---|---|
| (none) | 1 | 0 | 0 | 0 |
| 2019-04-17 | 0 | 10 | 0 | 0 |
| 2019-04-18 | 62 | 161 | 66 | 0 |
| 2019-04-19 | 951 | 536 | 61 | 468 |
| 2019-04-20 | 67 | 62 | 330 | 44 |
| 2019-04-21 | 45 | 4 | 33 | 0 |
| 2019-04-24 | 362 | 6 | 33 | 53 |

Most common capture day per class: healthy: 64% on 2019-04-19; leaf blast: 69% on 2019-04-19; brown spot: 63% on 2019-04-20; hispa: 83% on 2019-04-19. Where classes were photographed on different days, lighting and session conditions are partly tied to the class; this is reported, not corrected, and is one more reason the zero-shot test matters.

**JPEG quality** (the IJG quality whose luminance table matches; exact for 3,355 of 3,355 photos): quality 90: 1, quality 100: 3,354. Compression is practically uniform, so it cannot act as a shortcut. Different from the rest: `Healthy/IMG_20190419_145234.jpg` (quality 90, capture time missing).

## 6. Identical perceptual hashes (preview for Stage 2b)

13 groups of primary photos share an identical 64-bit perceptual hash (27 photos); 6 of them contain more than one class. 6 of these 6 were inspected by eye (2026-11-04): different leaves with the same layout (one leaf lying diagonally on a white background), not copies. Their photos were taken 3 min to 1 d 2 h 13 min apart. A hash match alone therefore means little on this dataset: the Stage 2b duplicate search will flag look-alike pairs, and the two-person review decides.

## 7. Paddy Doctor (held out; zero-shot only)

16,225 photos in 13 classes. Sizes: 1080 x 1440: 16,219, 1440 x 1080: 6. JPEG quality: quality 75: 16,225. Camera EXIF on 0 photos. The 4 shared classes hold 8,164 files: normal 2,405, blast 2,351, brown_spot 1,257, hispa 2,151. `metadata.csv` matches the files one to one (checked when the inventory is built; variety and age are joined).

## 8. Pixel scale: crops or resized photos?

Rule (pre-declared methodology): measure leaf width on 30 photos drawn across the size range; if Spearman's rho with the shorter side is >= 0.5, rescale every photo to the median leaf width before tiling, otherwise keep native resolution. The photos are one random pick per equal-width band of photo size on a log scale (seed 0), covering 734-3120 px, frozen in `data/leaf_width_selection.csv`. 2 measurers measure each photo independently, in their own shuffled order; at least 20 photos must be measurable (protocol in `src/leaf_width.py`). If rescaling is triggered, the median leaf width comes from these 30 photos, some of which are later test photos; it is a label-free geometry calibration declared in advance.

Spearman's rho = **0.539** (p = 0.00253, n = 29; 1 photos marked unmeasurable and left out), so: **rescale every photo to the median leaf width before tiling**. Median leaf width 195 px. The two measurers (ahsan, model_rater_a) agree with Spearman 0.98; their widths differ by 3.2% at the median (largest 35.6%); rho from each measurer alone: ahsan 0.542, model_rater_a 0.528. Leaving out one photo at a time gives rho 0.488-0.613; 5 of 29 single removals would flip the verdict. Model rater(s), not people: `model_rater_a` (an AI model measuring from rendered images with a coordinate grid, through the same tool code). How the rescale is applied: a log-log fit of leaf width on shorter side (slope 0.78) estimates each photo's leaf width, and the photo is scaled so that estimate becomes the median: factor 2.03 at 734 px to 0.66 at 3120 px (`rescale_factor` in `src/leaf_width.py`). Decision record: `data/pixel_scale_decision.json`.

| photo_id | short_side | ahsan | model_rater_a | leaf_width_px | used |
|---|---|---|---|---|---|
| LeafBlast/IMG_3169.jpg | 734 | 44.6 | 46.0 | 45.3 | True |
| BrownSpot/IMG_3177.jpg | 812 | – | 131.6 | 131.6 | False |
| LeafBlast/IMG_4953.jpg | 846 | 68.8 | 68.0 | 68.4 | True |
| Hispa/IMG_20190419_095315.jpg | 877 | 84.1 | 73.0 | 78.6 | True |
| Healthy/IMG_3203.jpg | 912 | 104.3 | 106.8 | 105.6 | True |
| LeafBlast/IMG_20190419_095029.jpg | 940 | 132.5 | 132.4 | 132.4 | True |
| LeafBlast/IMG_20190419_114718.jpg | 1000 | 270.6 | 268.0 | 269.3 | True |
| Hispa/IMG_20190424_131101.jpg | 1040 | 191.0 | 198.9 | 195.0 | True |
| Healthy/IMG_20190424_130136.jpg | 1085 | 328.5 | 336.4 | 332.4 | True |
| Hispa/IMG_20190419_171917.jpg | 1136 | 113.6 | 131.5 | 122.6 | True |
| BrownSpot/IMG_20190424_131042.jpg | 1197 | 245.4 | 264.1 | 254.8 | True |
| LeafBlast/IMG_20190424_133722.jpg | 1297 | 268.1 | 263.8 | 266.0 | True |
| Hispa/IMG_20190419_152515.jpg | 1348 | 205.8 | 199.3 | 202.6 | True |
| LeafBlast/IMG_20190419_112203.jpg | 1436 | 68.9 | 48.1 | 58.5 | True |
| Hispa/IMG_20190419_095917.jpg | 1478 | 150.1 | 141.5 | 145.8 | True |
| Hispa/IMG_20190419_165718.jpg | 1566 | 145.8 | 139.4 | 142.6 | True |
| Healthy/IMG_20190419_095521.jpg | 1664 | 194.7 | 193.0 | 193.8 | True |
| Healthy/IMG_20190419_100104.jpg | 1726 | 123.7 | 102.9 | 113.3 | True |
| Hispa/IMG_20190419_173037.jpg | 1799 | 132.3 | 149.0 | 140.6 | True |
| LeafBlast/IMG_3014.jpg | 1881 | 137.5 | 134.3 | 135.9 | True |
| Hispa/IMG_20190419_163245.jpg | 1976 | 210.6 | 217.2 | 213.9 | True |
| Hispa/IMG_20190419_161346.jpg | 2111 | 238.4 | 179.5 | 209.0 | True |
| Healthy/IMG_20190419_161852.jpg | 2155 | 153.8 | 142.0 | 147.9 | True |
| Healthy/IMG_20190420_194921.jpg | 2313 | 323.1 | 308.2 | 315.6 | True |
| Hispa/IMG_20190419_163101.jpg | 2404 | 261.3 | 264.6 | 263.0 | True |
| BrownSpot/IMG_20190420_191021.jpg | 2454 | 505.2 | 467.8 | 486.5 | True |
| Hispa/IMG_20190419_110519.jpg | 2616 | 224.6 | 219.9 | 222.2 | True |
| BrownSpot/IMG_20190420_195258.jpg | 2817 | 405.3 | 385.7 | 395.5 | True |
| Hispa/IMG_20190419_140037.jpg | 2895 | 221.7 | 221.6 | 221.6 | True |
| BrownSpot/IMG_20190419_135038.jpg | 3120 | 243.5 | 249.3 | 246.4 | True |

## 9. What this means for the next stages

- **Stage 2a (backgrounds):** the photos inspected so far show single leaves on plain white; the audit
  measures how general that is. If it holds, robustness to clutter cannot be claimed from in-domain
  results.
- **Stage 2b (near-duplicates):** expect hash matches between different leaves (section 6); the two-person
  review decides. Hash with `src.inventory.load_for_hash` and the `hashing` settings in `base.yaml`.
- **Stage 3 (splits):** use `photo_id` from `data/inventory.csv` as the key. For `splits/paddy4.csv`, follow
  section 3.
- **Stage 4b (shortcuts):** sections 4 and 5 show how size, camera and capture day are spread over the
  classes; the declared baselines (plus the camera-only one) measure what a model can get from them.
- **Stage 6 (tiling):** depends on the pixel-scale result (section 8).
