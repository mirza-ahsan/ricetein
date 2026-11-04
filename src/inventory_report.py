"""Stage 1 report: docs/reports/inventory_report.md, built from the inventories.

Every number is computed from the inventories. The only typed inputs are in ``configs/inventory.yaml``
under ``report`` (the published class counts and which hash groups were inspected by eye), and every
sentence that states a finding is only written when the numbers support it.

    uv run python -m src.inventory_report
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import dataset_dir, load_config, path_for
from src.inventory import read_inventory
from src.leaf_width import evaluate


# ---------------------------------------------------------------- formatting
def md_table(frame: pd.DataFrame) -> str:
    cells = frame.astype(object).where(frame.notna(), "–").astype(str)
    lines = ["| " + " | ".join(map(str, cells.columns)) + " |", "|" + "---|" * len(cells.columns)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in cells.to_numpy()])


def nice(name: str) -> str:
    return name.replace("_", " ")


def counts_text(counts: pd.Series, unit: str) -> str:
    """'quality 100: 3,354, quality 90: 1' style text for a value -> count series."""
    return ", ".join(f"{unit}{value}: {int(n):,}" for value, n in counts.items())


def percent(part: float, whole: float) -> str:
    share = part / whole if whole else 0.0
    return "<1%" if 0 < share < 0.005 else f"{share:.0%}"


def duration(delta: pd.Timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    if minutes == 0:
        return "under a minute"
    days, hours, mins = minutes // 1440, minutes % 1440 // 60, minutes % 60
    return " ".join(f"{n} {unit}" for n, unit in ((days, "d"), (hours, "h"), (mins, "min")) if n)


def duplicate_groups(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per value of ``column`` shared by 2+ files: how many files, and which folders."""
    shared = frame[frame[column].duplicated(keep=False) & frame[column].notna()]
    return shared.groupby(column).agg(files=("photo_id", "size"),
                                      folders=("folder", lambda s: " + ".join(sorted(set(s)))))


def verify_ignored_copies(config: dict[str, Any], primary: pd.DataFrame) -> tuple[int, int, int]:
    """In the ignored folders: (files, exact copies of an inventoried photo, those with the same class)."""
    spec = config["datasets"]["primary"]
    images = dataset_dir(config, "primary")
    root = images.parents[len(Path(spec["image_subdir"]).parts) - 1]  # the dataset folder itself
    folder_of = dict(zip(primary["sha256"], primary["folder"]))
    files = [p for d in spec["ignored_subdirs"] for p in (root / d).rglob("*") if p.is_file()]
    found = [folder_of.get(hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    same_class = sum(f == p.parent.name for f, p in zip(found, files))
    return len(files), sum(f is not None for f in found), same_class


# ---------------------------------------------------------------- sections
def counts_section(config: dict[str, Any], primary: pd.DataFrame, copies: tuple[int, int, int]) -> str:
    classes, spec = config["classes"], config["datasets"]["primary"]
    published = config["report"]["published_counts"]
    found = primary["class"].value_counts().reindex(classes, fill_value=0)
    table = pd.DataFrame({"class": [nice(c) for c in classes],
                          "folder": [spec["class_folders"][c] for c in classes],
                          "found": found.to_numpy(), "published (Do 2019)": [published[c] for c in classes]})
    table["difference"] = table["found"] - table["published (Do 2019)"]
    text = ("## 1. Primary dataset: class counts\n\n" + md_table(table) + "\n\n"
            f"Total **{len(primary):,}** files in `{spec['image_subdir']}/` (published: "
            f"{sum(published.values()):,}).")
    unexpected = [c for c in classes if found[c] != spec["expected_counts"][c]]
    if unexpected:
        text += f" **Warning:** {', '.join(map(nice, unexpected))} differ from `base.yaml`'s expected counts."
    extra = table[table["difference"] > 0]
    if len(extra) and not primary["sha256"].duplicated().any():
        listed = ", ".join(f"{r['class']} +{r['difference']}" for _, r in extra.iterrows())
        text += f" The extra photos ({listed}) are not exact copies of any other photo (section 3)."
    n_files, n_copies, n_same = copies
    if n_files:
        folders = ", ".join(f"`{d}/`" for d in spec["ignored_subdirs"])
        text += (f"\n\nThe download's other folder(s) ({folders}) hold {n_files:,} files: {n_copies:,} are "
                 f"byte-identical copies of inventoried photos and {n_same:,} of those carry the same class.")
        missing = " **Not all are copies: these photos are missing from the inventory.**"
        text += (" They add no photos, so they are ignored and no photo is counted twice."
                 if n_copies == n_files else missing)
    return text


def integrity_section(primary: pd.DataFrame, paddy: pd.DataFrame) -> str:
    frames = {"primary": primary, "Paddy Doctor": paddy}
    rows = [{"dataset": name, "files": f"{len(f):,}", "decode fully": f"{int(f['opened_ok'].sum()):,}",
             "formats": ", ".join(f["format"].dropna().unique()),
             "modes": ", ".join(f["mode"].dropna().unique()),
             "rotation tags": int((f["exif_orientation"].fillna(1) != 1).sum())}
            for name, f in frames.items()]
    both = pd.concat(frames.values())
    broken = both[~both["opened_ok"]]
    facts = ["Every file decodes fully" if broken.empty else
             f"**{len(broken)} file(s) fail to decode**: "
             + ", ".join(f"`{r.photo_id}` ({r.error})" for r in broken.itertuples())]
    facts.append(f"formats: {', '.join(sorted(set(both['format'].dropna())))}; colour modes: "
                 f"{', '.join(sorted(set(both['mode'].dropna())))}")
    rotated = int((both["exif_orientation"].fillna(1) != 1).sum())
    facts.append("no file carries a rotation tag, so photos are read as stored and orientation handling "
                 "cannot differ between machines" if rotated == 0 else
                 f"{rotated} files carry a rotation tag (their sizes and hashes are recorded upright)")
    return "## 2. File integrity\n\n" + md_table(pd.DataFrame(rows)) + "\n\n" + "; ".join(facts) + "."


def duplicates_section(config: dict[str, Any], primary: pd.DataFrame, paddy: pd.DataFrame) -> str:
    pdup, qdup = duplicate_groups(primary, "sha256"), duplicate_groups(paddy, "sha256")
    shared_files = len(set(primary["sha256"].dropna()) & set(paddy["sha256"].dropna()))
    lines = [f"- **Primary:** {len(pdup)} duplicate groups.",
             f"- **Between the datasets:** {shared_files} shared files."]
    if qdup.empty:
        lines.append("- **Paddy Doctor:** 0 duplicate groups.")
        return "## 3. Exact duplicates (same SHA-256)\n\n" + "\n".join(lines)
    shared = set(config["datasets"]["paddy"]["class_folders"].values())
    in_shared = paddy[paddy["sha256"].isin(qdup.index) & paddy["folder"].isin(shared)]
    conflicts = qdup[qdup["folders"].str.contains("+", regex=False)]
    conflict_files = paddy[paddy["sha256"].isin(conflicts.index)]
    test_conflicts = conflict_files[conflict_files["folder"].isin(shared)]
    meta = paddy[paddy["sha256"].isin(qdup.index)].groupby("sha256")[["variety", "age"]].nunique()
    sizes = counts_text(qdup["files"].value_counts().sort_index(), "groups of ")
    line = (f"- **Paddy Doctor:** {len(qdup)} groups covering {int(qdup['files'].sum()):,} files ({sizes}). "
            f"In the 4 classes used for zero-shot testing: {len(in_shared)} files in "
            f"{in_shared['sha256'].nunique()} groups, i.e. {len(in_shared) - in_shared['sha256'].nunique()} "
            "extra copies.")
    if len(conflicts):
        listed = ", ".join(f"`{p}`" for p in conflict_files["photo_id"])
        where = (f"{len(test_conflicts)} of these files are in the 4 test classes" if len(test_conflicts)
                 else "none of them is in the 4 test classes")
        line += f" {len(conflicts)} group(s) span more than one class ({listed}); {where}."
    if (meta.max(axis=1) == 1).all():
        line += (" Every group has the same variety and age in `metadata.csv`, so these look like files "
                 "stored twice in the release, not separate photos.")
    todo = "count each image once" + (" and drop images filed under two test classes" if len(test_conflicts)
                                       else "")
    lines += [line, f"\n**To decide in Stage 3** (`splits/paddy4.csv`): {todo}."]
    return "## 3. Exact duplicates (same SHA-256)\n\n" + "\n".join(lines)


def sizes_section(config: dict[str, Any], primary: pd.DataFrame) -> str:
    largest = int(primary["short_side"].max())

    def at_largest(sides: pd.Series) -> str:
        return percent((sides == largest).sum(), len(sides))

    sizes = primary.groupby("class").agg(
        photos=("photo_id", "size"), shortest=("short_side", "min"),
        median=("short_side", lambda s: int(s.median())),
        longest=("short_side", "max"), at_largest=("short_side", at_largest),
        **{"median MB": ("file_bytes", lambda s: round(s.median() / 1e6, 2))},
    ).reindex(config["classes"]).rename(columns={"at_largest": f"at {largest} px"})
    med = sizes["median"].dropna()
    aspect = (primary[["width", "height"]].max(axis=1) / primary["short_side"] - 1).max()
    shape = (f"Photos are square or nearly so (longer side at most {aspect:.2%} longer than the shorter)"
             if aspect < 0.01 else f"Photos are not all square (longer side up to {aspect:.0%} longer)")
    return ("## 4. Photo size by class (a possible shortcut)\n\nShorter side in pixels.\n\n"
            + md_table(sizes.rename(index=nice).rename_axis("class").reset_index()) + "\n\n"
            f"{shape}; the shorter side runs from {int(primary['short_side'].min())} to {largest} px (median "
            f"{int(primary['short_side'].median())}). The class with the smallest photos is "
            f"**{nice(med.idxmin())}** (median {int(med.min())} px, against {int(med.max())} px for "
            f"{nice(med.idxmax())}). A model could pick up size instead of disease; the photo-properties "
            "baseline (Stage 4b) measures how much.")


def camera_text(classes: list[str], primary: pd.DataFrame) -> str:
    cams = pd.crosstab(primary["class"], primary["camera"]).reindex(classes, fill_value=0)
    if not cams.shape[1]:
        return "**Camera:** no photo carries camera EXIF."
    totals = cams.sum(axis=1)
    cells = cams.apply(lambda col: [f"{n:,} ({percent(n, t)})" for n, t in zip(col, totals)])
    main = cams.sum().idxmax()
    others = ", ".join(f"{nice(c)} {percent(totals[c] - cams.loc[c, main], totals[c])}" for c in classes)
    table = md_table(cells.rename(index=nice).rename_axis("class").reset_index())
    return (f"**Camera (EXIF make and model).**\n\n{table}\n\n{cams.shape[1]} cameras appear. Share of "
            "photos "
            f"not taken with the most common one ({main}): {others}. A model could pick up the camera "
            "instead "
            "of the disease; Stage 4b adds a declared camera-only baseline next to the methodology's "
            "baselines.")


def capture_day_text(classes: list[str], primary: pd.DataFrame) -> str:
    day = primary["taken_at"].astype("string").str[:10].str.replace(":", "-")
    if not day.notna().any():
        return "**Capture day:** no photo carries a capture time."
    days = pd.crosstab(day.fillna("(none)"), primary["class"]).reindex(columns=classes, fill_value=0)
    modal = "; ".join(f"{nice(c)}: {percent(days[c].max(), days[c].sum())} on {days[c].idxmax()}"
                      for c in classes if days[c].sum())
    table = md_table(days.rename(columns=nice).rename_axis("capture day").reset_index())
    return (f"**Capture day (EXIF DateTimeOriginal).**\n\n{table}\n\nMost common capture day per class: "
            f"{modal}. "
            "Where classes were photographed on different days, lighting and session conditions are partly "
            "tied to the class; this is reported, not corrected, and is one more reason the zero-shot test "
            "matters.")


def quality_text(primary: pd.DataFrame) -> str:
    quality = primary["jpeg_quality"].dropna().astype(int).value_counts().sort_index()
    if quality.empty:
        return "**JPEG quality:** not available (no JPEG files)."
    exact = int(primary["jpeg_quality_exact"].fillna(False).astype(bool).sum())
    text = (f"**JPEG quality** (the IJG quality whose luminance table matches; exact for {exact:,} of "
            f"{len(primary):,} photos): {counts_text(quality, 'quality ')}.")
    main = quality.idxmax()
    if quality[main] / quality.sum() > 0.99:
        text += " Compression is practically uniform, so it cannot act as a shortcut."
    odd = primary[primary["jpeg_quality"].notna() & (primary["jpeg_quality"] != main)]
    if 0 < len(odd) <= 5:
        text += " Different from the rest: " + ", ".join(
            f"`{r.photo_id}` (quality {r.jpeg_quality}, capture time "
            f"{'missing' if pd.isna(r.taken_at) else r.taken_at})" for r in odd.itertuples()) + "."
    return text


def camera_section(config: dict[str, Any], primary: pd.DataFrame) -> str:
    classes = config["classes"]
    return "\n\n".join(["## 5. Camera, capture day and JPEG quality by class", camera_text(classes, primary),
                        capture_day_text(classes, primary), quality_text(primary)])


def phash_section(config: dict[str, Any], primary: pd.DataFrame) -> str:
    groups = duplicate_groups(primary, "phash")
    n_classes = primary.groupby("phash")["class"].nunique().reindex(groups.index)
    cross = groups[n_classes > 1]
    text = ("## 6. Identical perceptual hashes (preview for Stage 2b)\n\n"
            f"{len(groups)} groups of primary photos share an identical 64-bit perceptual hash "
            f"({int(groups['files'].sum())} photos); {len(cross)} of them contain more than one class.")
    if cross.empty:
        return text
    inspected = config["report"]["inspected_phash_groups"]
    seen = cross.index.intersection(inspected["hashes"])
    times = pd.to_datetime(primary["taken_at"], format="%Y:%m:%d %H:%M:%S", errors="coerce")
    spans = times.groupby(primary["phash"]).agg(lambda t: t.max() - t.min()).reindex(cross.index).dropna()
    text += (f" {len(seen)} of these {len(cross)} were inspected by eye ({inspected['date']}): "
             f"{inspected['verdict']}, not copies.")
    if len(spans):
        text += f" Their photos were taken {duration(spans.min())} to {duration(spans.max())} apart."
    return text + (" A hash match alone therefore means little on this dataset: the Stage 2b duplicate "
                   "search "
                   "will flag look-alike pairs, and the two-person review decides.")


def paddy_section(config: dict[str, Any], paddy: pd.DataFrame) -> str:
    per_folder = paddy["folder"].value_counts()
    dims = (paddy["width"].astype(str) + " x " + paddy["height"].astype(str)).value_counts()
    folders = config["datasets"]["paddy"]["class_folders"].values()
    shared = ", ".join(f"{f} {int(per_folder.get(f, 0)):,}" for f in folders)
    quality = paddy["jpeg_quality"].dropna().astype(int).value_counts().sort_index()
    return ("## 7. Paddy Doctor (held out; zero-shot only)\n\n"
            f"{len(paddy):,} photos in {paddy['folder'].nunique()} classes. Sizes: {counts_text(dims, '')}. "
            f"JPEG quality: {counts_text(quality, 'quality ')}. Camera EXIF on "
            f"{int(paddy['camera'].notna().sum()):,} photos. The 4 shared classes hold "
            f"{int(paddy['class'].notna().sum()):,} files: {shared}. `metadata.csv` matches the files "
            "one to one "
            "(checked when the inventory is built; variety and age are joined).")


def pixel_scale_section(config: dict[str, Any]) -> str:
    settings = config["pixel_scale"]
    result = evaluate(config)
    sides = result["selected"]["short_side"]
    text = ("## 8. Pixel scale: crops or resized photos?\n\n"
            f"Rule (pre-declared methodology): measure leaf width on {settings['n_photos']} photos drawn "
            "across the "
            f"size range; if Spearman's rho with the shorter side is >= {settings['spearman_threshold']}, "
            "rescale every photo to the median leaf width before tiling, otherwise keep native resolution. "
            "The photos are one random pick per equal-width band of photo size on a log scale (seed "
            f"{config['seed']}), covering {int(sides.min())}-{int(sides.max())} px, frozen in "
            f"`{settings['selection_file']}`. {settings['measurers']} team members measure each photo "
            f"independently, in their own shuffled order; at least {settings['min_usable']} photos must be "
            "measurable (protocol in `src/leaf_width.py`). If rescaling is triggered, the median leaf width "
            f"comes from these {len(sides)} photos, some of which are later test photos; it is a label-free "
            "geometry calibration declared in advance.\n\n")
    if not result["complete"]:
        progress = ", ".join(f"{who} {n}/{len(sides)}" for who, n in result["answered"].items())
        return text + (f"**Pending.** Answered so far: {progress or 'nobody yet'}. Each measurer runs "
                       "`uv run python -m src.leaf_width measure --measurer <name>`; then "
                       "`uv run python -m src.leaf_width decide`.")
    columns = ["photo_id", "short_side", *result["measurers"], "leaf_width_px", "used"]
    table = result["table"][columns].round(1)
    if not result["applied"]:
        return text + f"**Rule not applied:** {result['reason']}.\n\n" + md_table(table)
    verdict = ("**rescale every photo to the median leaf width before tiling**" if result["rescale"]
               else "**keep native resolution** (photos behave like crops at the camera's own scale)")
    agreement = result["agreement_spearman"]
    return text + (
        f"Spearman's rho = **{result['spearman_rho']:.3f}** (p = {result['p_value']:.3g}, "
        f"n = {result['n_used']}; {result['n_unmeasurable']} photos marked unmeasurable and left out), so: "
        f"{verdict}. Median leaf width {result['median_leaf_width_px']:.0f} px. The two measurers "
        f"({', '.join(result['measurers'])}) agree with Spearman "
        f"{'n/a' if agreement is None else f'{agreement:.2f}'}; their widths differ by "
        f"{result['median_relative_gap']:.1%} at the median (largest {result['max_relative_gap']:.1%}). "
        f"Decision record: `{settings['decision_file']}`.\n\n" + md_table(table))


NEXT_STAGES = """## 9. What this means for the next stages

- **Stage 2a (backgrounds):** the photos inspected so far show single leaves on plain white; the audit
  measures how general that is. If it holds, robustness to clutter cannot be claimed from in-domain
  results.
- **Stage 2b (near-duplicates):** expect hash matches between different leaves (section 6); the two-person
  review decides. Hash with `src.inventory.load_for_hash` and the `hashing` settings in `base.yaml`.
- **Stage 3 (splits):** use `photo_id` from `data/inventory.csv` as the key. For `splits/paddy4.csv`, follow
  section 3.
- **Stage 4b (shortcuts):** sections 4 and 5 show how size, camera and capture day are spread over the
  classes; the declared baselines (plus the camera-only one) measure what a model can get from them.
- **Stage 6 (tiling):** depends on the pixel-scale result (section 8)."""


def build_report(config: dict[str, Any]) -> str:
    files = config["inventory"]["files"]
    primary, paddy = read_inventory(files["primary"]), read_inventory(files["paddy"])
    for frame in (primary, paddy):
        frame["short_side"] = frame[["width", "height"]].min(axis=1)
    header = ("# Stage 1 inventory report\n\nBuilt by `src/inventory_report.py` from "
              f"`{files['primary']}` and `{files['paddy']}`. Every number is computed; the only typed inputs "
              "are the published counts and the record of which hash groups were inspected by eye "
              "(`configs/inventory.yaml`, `report`).")
    sections = [
        header, counts_section(config, primary, verify_ignored_copies(config, primary)),
        integrity_section(primary, paddy), duplicates_section(config, primary, paddy),
        sizes_section(config, primary), camera_section(config, primary), phash_section(config, primary),
        paddy_section(config, paddy), pixel_scale_section(config), NEXT_STAGES,
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
