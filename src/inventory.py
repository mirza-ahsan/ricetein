"""Stage 1 inventory: one row per photo, with everything later stages need to know about it.

For every file: a stable ID (its path inside the dataset, e.g. ``Hispa/IMG_20190419_172808.jpg``),
class, size, format, SHA-256, 64-bit perceptual hash, whether it opens, EXIF orientation, camera,
capture time and estimated JPEG quality. Camera and JPEG quality are recorded because they can
differ by class, which a model could exploit (handbook Stage 1 step 7 and Stage 4b).

    uv run python -m src.inventory                 # both datasets -> data/*.csv + report
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.config import REPO_ROOT, dataset_dir, load_config

# Luminance quantisation table of the JPEG standard (Annex K), the reference for quality 50.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
])


def jpeg_quality(image: Image.Image) -> float | None:
    """Estimated IJG quality (1-100) from the luminance table; None if not a JPEG."""
    tables = getattr(image, "quantization", None)
    if not tables or 0 not in tables:
        return None
    scale = float(np.mean(np.asarray(tables[0], dtype=float) * 100 / _STD_LUMA))  # IJG scaling, inverted
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return round(min(max(quality, 1.0), 100.0), 1)


def load_for_hash(path: Path, draft_size: int) -> Image.Image:
    """The exact decode used for perceptual hashes (Stage 2 must hash the same way)."""
    image = Image.open(path)
    image.draft("RGB", (draft_size, draft_size))  # JPEG: decode at a reduced scale, much faster
    return ImageOps.exif_transpose(image).convert("RGB")


def describe_file(path: Path, root: Path, hash_size: int, draft_size: int) -> dict[str, Any]:
    """Everything recorded about one file. Never raises: failures are recorded instead."""
    row: dict[str, Any] = {"photo_id": path.relative_to(root).as_posix(), "file_bytes": path.stat().st_size}
    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            details = exif.get_ifd(0x8769)
            row.update({
                "width": image.width, "height": image.height, "format": image.format, "mode": image.mode,
                "exif_orientation": exif.get(0x0112, 1),
                "camera": " ".join(str(exif.get(t, "")).strip() for t in (0x010F, 0x0110)).strip() or None,
                "taken_at": details.get(0x9003),  # DateTimeOriginal
                "jpeg_quality": jpeg_quality(image),
            })
            image.load()  # forces a full decode: truncated or corrupt files fail here
        row["phash"] = str(imagehash.phash(load_for_hash(path, draft_size), hash_size=hash_size))
        row["opened_ok"], row["error"] = True, None
    except Exception as exc:  # recorded in the inventory, reported, never silently skipped
        row["opened_ok"], row["error"] = False, f"{type(exc).__name__}: {exc}"
    return row


def _describe(args: tuple[Path, Path, int, int]) -> dict[str, Any]:
    return describe_file(*args)


def scan(root: Path, hash_size: int, draft_size: int, workers: int | None = None) -> pd.DataFrame:
    """Describe every file under ``root`` (all files, not only images), sorted by ID."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    jobs = [(p, root, hash_size, draft_size) for p in files]
    # spawn, not fork: forking a process that already runs threads (torch, Jupyter) can deadlock
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count(), mp_context=spawn) as pool:
        rows = list(pool.map(_describe, jobs, chunksize=16))
    frame = pd.DataFrame(rows)
    in_folder = frame["photo_id"].str.contains("/")
    frame.insert(1, "folder", frame["photo_id"].str.split("/").str[0].where(in_folder))
    return frame


def primary_inventory(config: dict[str, Any], workers: int | None = None) -> pd.DataFrame:
    spec, inv = config["datasets"]["primary"], config["inventory"]
    frame = scan(dataset_dir(config, "primary"), inv["hash_size"], inv["hash_draft_size"], workers)
    to_class = {folder: name for name, folder in spec["class_folders"].items()}
    frame.insert(2, "class", frame["folder"].map(to_class))
    return frame


def paddy_inventory(config: dict[str, Any], workers: int | None = None) -> pd.DataFrame:
    """All 13 Paddy Doctor classes; ``class`` is set only for the 4 shared with the primary dataset."""
    spec, inv = config["datasets"]["paddy"], config["inventory"]
    root = dataset_dir(config, "paddy")
    frame = scan(root, inv["hash_size"], inv["hash_draft_size"], workers)
    frame = frame[frame["photo_id"] != spec["metadata_file"]].reset_index(drop=True)
    to_class = {folder: name for name, folder in spec["class_folders"].items()}
    frame.insert(2, "class", frame["folder"].map(to_class))
    meta = pd.read_csv(root / spec["metadata_file"]).rename(columns={"label": "folder"})
    meta["photo_id"] = meta["folder"] + "/" + meta["image_id"]
    return frame.merge(meta[["photo_id", "variety", "age"]], on="photo_id", how="left", validate="1:1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/inventory.csv and data/paddy_inventory.csv.")
    parser.add_argument("--config", default="configs/inventory.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    workers = config["inventory"]["workers"]
    builds = (("primary", primary_inventory, "primary_file"), ("paddy", paddy_inventory, "paddy_file"))
    for name, build, key in builds:
        frame = build(config, workers)
        target = REPO_ROOT / config["inventory"][key]
        frame.to_csv(target, index=False)
        print(f"{name}: {len(frame)} files -> {target.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
