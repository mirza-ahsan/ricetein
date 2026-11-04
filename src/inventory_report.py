"""Stage 1 report: docs/reports/inventory_report.md, built from the inventories (no typed numbers).

    uv run python -m src.inventory_report
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date
from typing import Any

import pandas as pd

from src.config import dataset_dir, load_config, path_for
from src.inventory import read_inventory
from src.leaf_width import decide

PUBLISHED_PRIMARY = {"healthy": 1488, "leaf_blast": 779, "brown_spot": 523, "hispa": 563}  # Do (2019)


def md_table(frame: pd.DataFrame) -> str:
    cells = frame.astype(str)
    lines = ["| " + " | ".join(cells.columns) + " |", "|" + "---|" * len(cells.columns)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in cells.to_numpy()])


def nice(name: str) -> str:
    return name.replace("_", " ")


def duplicate_groups(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per value of ``column`` shared by 2+ files: how many files, and which folders."""
    shared = frame[frame[column].duplicated(keep=False)]
    return shared.groupby(column).agg(files=("photo_id", "size"),
                                      folders=("folder", lambda s: " + ".join(sorted(set(s)))))


def verify_split_copy(config: dict[str, Any], primary: pd.DataFrame) -> tuple[int, int, int]:
    """In the ignored RiceDiseaseDataset/: (files, exact copies of a Labelled photo, of those same class)."""
    spec = config["datasets"]["primary"]
    root = dataset_dir(config, "primary").parents[1]  # dataset root, above LabelledRice/Labelled
    folder_of = dict(zip(primary["sha256"], primary["folder"]))
    files = [p for d in spec["ignored_subdirs"] for p in (root / d).rglob("*") if p.is_file()]
    found = [folder_of.get(hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    same_class = sum(f == p.parent.name for f, p in zip(found, files))
    return len(files), sum(f is not None for f in found), same_class


# ---------------------------------------------------------------- sections
def counts_section(config: dict[str, Any], primary: pd.DataFrame) -> str:
    classes = config["classes"]
    folders = config["datasets"]["primary"]["class_folders"]
    found = primary["class"].value_counts()
    table = pd.DataFrame({
        "class": [nice(c) for c in classes], "folder": [folders[c] for c in classes],
        "found": [int(found.get(c, 0)) for c in classes],
        "published (Do 2019)": [PUBLISHED_PRIMARY[c] for c in classes],
    })
    table["difference"] = table["found"] - table["published (Do 2019)"]
    n_files, n_copies, n_same = verify_split_copy(config, primary)
    return (
        "## 1. Primary dataset: class counts\n\n" + md_table(table) + "\n\n"
        f"Total **{len(primary):,}** files in `LabelledRice/Labelled/` (published: "
        f"{sum(PUBLISHED_PRIMARY.values()):,}). The download's second folder (`RiceDiseaseDataset/`) holds "
        f"{n_files:,} files: {n_copies:,} are byte-identical copies of `Labelled/` photos and {n_same:,} "
        "carry the same class, so it is ignored and no photo is counted twice. The extra Hispa photos are "
        "not exact copies of any other photo (section 3)."
    )


def integrity_section(primary: pd.DataFrame, paddy: pd.DataFrame) -> str:
    rows = [{
        "dataset": name, "files": f"{len(frame):,}", "decode fully": f"{int(frame['opened_ok'].sum()):,}",
        "formats": ", ".join(frame["format"].dropna().unique()),
        "modes": ", ".join(frame["mode"].dropna().unique()),
        "rotation tags": int((frame["exif_orientation"].fillna(1) != 1).sum()),
    } for name, frame in (("primary", primary), ("Paddy Doctor", paddy))]
    return (
        "## 2. File integrity\n\n" + md_table(pd.DataFrame(rows)) + "\n\n"
        "No file is broken, every file is an RGB JPEG and none carries a rotation tag, so photos are read "
        "as stored and orientation handling cannot differ between machines."
    )


def duplicates_section(config: dict[str, Any], primary: pd.DataFrame, paddy: pd.DataFrame) -> str:
    pdup, qdup = duplicate_groups(primary, "sha256"), duplicate_groups(paddy, "sha256")
    conflicts = qdup[qdup["folders"].str.contains(r"\+")]
    conflict_ids = ", ".join(f"`{p}`" for p in paddy.loc[paddy["sha256"].isin(conflicts.index), "photo_id"])
    shared = set(config["datasets"]["paddy"]["class_folders"].values())
    in_shared = paddy[paddy["sha256"].isin(qdup.index) & paddy["folder"].isin(shared)]
    same_meta = paddy[paddy["sha256"].isin(qdup.index)].groupby("sha256")[["variety", "age"]].nunique()
    sizes = qdup["files"].value_counts().sort_index().to_dict()
    return (
        "## 3. Exact duplicates (same SHA-256)\n\n"
        f"- **Primary:** {len(pdup)} duplicate groups.\n"
        f"- **Between the datasets:** {len(set(primary['sha256']) & set(paddy['sha256']))} shared files.\n"
        f"- **Paddy Doctor:** {len(qdup)} groups covering {int(qdup['files'].sum())} files (group sizes "
        f"{sizes}); {len(conflicts)} group(s) span two classes ({conflict_ids}). In the 4 classes used for "
        f"zero-shot testing, {len(in_shared)} files fall in {in_shared['sha256'].nunique()} groups, i.e. "
        f"{len(in_shared) - in_shared['sha256'].nunique()} extra copies. "
        f"{int((same_meta.max(axis=1) == 1).sum())} of {len(same_meta)} groups share the same variety "
        "and age "
        "in `metadata.csv`, so these look like files stored twice in the release, not separate photos.\n\n"
        "**To decide in Stage 3** (`splits/paddy4.csv`): count each image once and drop label conflicts."
    )


def sizes_section(config: dict[str, Any], primary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    sizes = primary.groupby("class").agg(
        photos=("photo_id", "size"), short_min=("short_side", "min"), short_median=("short_side", "median"),
        short_max=("short_side", "max"), share_3120=("short_side", lambda s: f"{(s >= 3120).mean():.0%}"),
        MB_median=("file_bytes", lambda s: round(s.median() / 1e6, 2)),
    ).reindex(config["classes"])
    med = sizes["short_median"]
    table = md_table(sizes.rename(index=nice).reset_index())
    text = (
        "## 4. Photo size by class (a possible shortcut)\n\n" + table
        + f"\n\nPhotos are (nearly) square, with the shorter side from {int(primary['short_side'].min())} to "
        f"{int(primary['short_side'].max())} px (median {int(primary['short_side'].median())}). The "
        "class with "
        f"the smallest photos is **{nice(med.idxmin())}** (median {int(med.min())} px, against "
        f"{int(med.max())} px for {nice(med.idxmax())}). A model could pick up size instead of disease; the "
        "photo-properties baseline (Stage 4b) measures how much."
    )
    return text, sizes


def camera_section(config: dict[str, Any], primary: pd.DataFrame) -> str:
    classes = config["classes"]
    cams = pd.crosstab(primary["class"], primary["camera"].fillna("(none)")).reindex(classes)
    share = cams.div(cams.sum(axis=1), axis=0).map(lambda v: f"{v:.0%}")
    cam_table = (cams.astype(str) + " (" + share + ")").rename(index=nice).reset_index()
    main_phone = cams.sum().idxmax()
    not_main = ", ".join(f"{nice(c)} {1 - cams.loc[c, main_phone] / cams.loc[c].sum():.0%}" for c in classes)
    day = primary["taken_at"].str[:10].str.replace(":", "-").fillna("(none)")
    days = pd.crosstab(day, primary["class"])[classes]
    modal = "; ".join(f"{nice(c)} {days[c].max() / days[c].sum():.0%} on {days[c].idxmax()}" for c in classes)
    quality = primary["jpeg_quality"].value_counts().sort_index()
    odd = primary[primary["jpeg_quality"] != quality.idxmax()]
    odd_text = ", ".join(
        f"`{r.photo_id}` (quality {r.jpeg_quality}, capture time "
        f"{'missing' if pd.isna(r.taken_at) else r.taken_at})" for r in odd.itertuples())
    return (
        "## 5. Camera, capture day and JPEG quality by class\n\n**Camera (EXIF make and model).**\n\n"
        + md_table(cam_table) + "\n\n"
        f"{cams.shape[1]} phones were used and their mix differs by class. Share of photos not taken with "
        f"the main phone ({main_phone}): {not_main}. Like size, the camera is a possible shortcut; Stage 4b "
        "adds a declared camera-only baseline next to the methodology's baselines.\n\n"
        "**Capture day (EXIF DateTimeOriginal).**\n\n" + md_table(days.rename(columns=nice).reset_index())
        + f"\n\nClasses were photographed on different days in different proportions ({modal}), so "
        "lighting and session conditions are partly tied to the class. This is reported, not corrected; it "
        "is one more reason the zero-shot test matters.\n\n"
        f"**JPEG quality** (the IJG quality whose luminance table matches; exact for "
        f"{int(primary['jpeg_quality_exact'].sum()):,} of {len(primary):,} photos): {quality.to_dict()} "
        "photos "
        f"per quality, so compression is not a shortcut. The odd one out: {odd_text or 'none'}; with no "
        "capture time it was most likely re-saved by an editor."
    )


def phash_section(primary: pd.DataFrame) -> str:
    groups = duplicate_groups(primary, "phash")
    classes = primary.groupby("phash")["class"].agg(lambda s: " + ".join(sorted(set(s))))
    cross = int(classes.reindex(groups.index).str.contains(r"\+").sum())
    return (
        "## 6. Identical perceptual hashes (preview for Stage 2b)\n\n"
        f"{len(groups)} groups of primary photos share an identical 64-bit perceptual hash "
        f"({int(groups['files'].sum())} photos; {cross} groups span two classes). All cross-class "
        "groups were "
        "inspected by eye (2026-11-04): each is **different leaves with the same layout** (one leaf lying "
        "diagonally on a white background, taken minutes to a day apart), not copies. So on this dataset a "
        "hash match alone means little: the Stage 2b duplicate search will flag many look-alike pairs, and "
        "the two-person review is what decides."
    )


def paddy_section(config: dict[str, Any], paddy: pd.DataFrame) -> str:
    per_folder = paddy["folder"].value_counts()
    dims = (paddy["width"].astype(str) + " x " + paddy["height"].astype(str)).value_counts().to_dict()
    folders = config["datasets"]["paddy"]["class_folders"].values()
    shared = ", ".join(f"{f} {per_folder[f]:,}" for f in folders)
    return (
        "## 7. Paddy Doctor (held out; zero-shot only)\n\n"
        f"{len(paddy):,} photos in {paddy['folder'].nunique()} classes; sizes {dims}; JPEG quality "
        f"{sorted(paddy['jpeg_quality'].dropna().unique().tolist())} (exact match for "
        f"{int(paddy['jpeg_quality_exact'].sum()):,}); no camera EXIF. The 4 shared classes hold "
        f"{int(paddy['class'].notna().sum()):,} files: {shared}. `metadata.csv` matches the files one to one "
        "(variety and age are joined into the inventory)."
    )


def pixel_scale_section(config: dict[str, Any]) -> str:
    settings = config["pixel_scale"]
    result = decide(config)
    sides = result["selected"]["short_side"]
    text = (
        "## 8. Pixel scale: crops or resized photos?\n\n"
        f"Rule (pre-declared methodology): measure leaf width on {settings['n_photos']} photos drawn across the "
        f"size range; if Spearman's rho with the shorter side is >= {settings['spearman_threshold']}, "
        "rescale "
        "every photo to the median leaf width before tiling, otherwise keep native resolution. The "
        "photos are "
        f"one random pick per equal-width band of photo size on a log scale (seed {config['seed']}), "
        "covering "
        f"{int(sides.min())}-{int(sides.max())} px, frozen in `{settings['selection_file']}`. "
        f"{settings['measurers']} team members measure each photo independently, in their own shuffled "
        f"order; at least {settings['min_usable']} photos must be measurable (protocol in "
        "`src/leaf_width.py`).\n\n"
    )
    if not result["complete"]:
        progress = ", ".join(f"{who} {n}/{len(sides)}" for who, n in result["answered"].items())
        return text + (f"**Pending.** Answered so far: {progress or 'nobody yet'}. Each measurer runs "
                       "`uv run python -m src.leaf_width measure --measurer <name>`; then "
                       "`uv run python -m src.leaf_width decide`.")
    verdict = ("**rescale every photo to the median leaf width before tiling**" if result["rescale"]
               else "**keep native resolution** (photos behave like crops at the camera's own scale)")
    agreement = result["agreement_spearman"]
    table = result["table"][["photo_id", "short_side", *result["measurers"], "leaf_width_px", "used"]]
    return text + (
        f"Spearman's rho = **{result['spearman_rho']:.3f}** (p = {result['p_value']:.3g}, "
        f"n = {result['n_used']}; {result['n_unmeasurable']} photos marked unmeasurable and left out), so: "
        f"{verdict}. Median leaf width {result['median_leaf_width_px']:.0f} px. The two measurers "
        f"({', '.join(result['measurers'])}) agree with Spearman "
        f"{'n/a' if agreement is None else f'{agreement:.2f}'}; their widths differ by "
        f"{result['median_relative_gap']:.1%} at the median (largest {result['max_relative_gap']:.1%}). "
        f"Decision record: `{settings['decision_file']}`.\n\n" + md_table(table.round(1))
    )


NEXT_STAGES = """## 9. What this means for the next stages

- **Stage 2a (backgrounds):** the photos looked at so far show single leaves on plain white; the audit
  measures how general that is. If it holds, robustness to clutter cannot be claimed from in-domain
  results.
- **Stage 2b (near-duplicates):** expect many hash matches between different leaves; the two-person review
  decides. Hash with `src.inventory.load_for_hash` and the `hashing` settings in `base.yaml`.
- **Stage 3 (splits):** use `photo_id` from `data/inventory.csv` as the key. For `splits/paddy4.csv`,
  count each Paddy Doctor image once and drop the label conflict (section 3).
- **Stage 4b (shortcuts):** photo size, camera and capture day all differ by class. The declared
  baselines cover size; a camera-only baseline is added (handbook decision 2026-11-04).
- **Stage 6 (tiling):** depends on the pixel-scale result (section 8)."""


def build_report(config: dict[str, Any]) -> str:
    files = config["inventory"]["files"]
    primary, paddy = read_inventory(files["primary"]), read_inventory(files["paddy"])
    for frame in (primary, paddy):
        frame["short_side"] = frame[["width", "height"]].min(axis=1)
    header = (f"# Stage 1 inventory report\n\nBuilt by `src/inventory_report.py` on "
              f"{date.today().isoformat()} from `{files['primary']}` and `{files['paddy']}`. Every number "
              "below is computed, not typed.")
    sections = [
        header, counts_section(config, primary), integrity_section(primary, paddy),
        duplicates_section(config, primary, paddy), sizes_section(config, primary)[0],
        camera_section(config, primary), phash_section(primary), paddy_section(config, paddy),
        pixel_scale_section(config), NEXT_STAGES,
    ]
    return "\n\n".join(sections) + "\n"


def write_report(config: dict[str, Any]) -> str:
    """Build the report, write docs/reports/inventory_report.md and return its text."""
    text = build_report(config)
    target = path_for(config, "reports") / "inventory_report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write docs/reports/inventory_report.md.")
    parser.add_argument("--config", default="configs/inventory.yaml")
    write_report(load_config(parser.parse_args().config))
    print("wrote docs/reports/inventory_report.md")


if __name__ == "__main__":
    main()
