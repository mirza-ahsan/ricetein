"""Stage 1 inventory: one row per file, with everything later stages need to know about it.

For every file under a dataset (not only images): a stable ID (its path inside the dataset, e.g.
``Hispa/IMG_20190419_172808.jpg``), class, upright size, format, SHA-256, 64-bit perceptual hash,
whether it decodes fully, EXIF orientation, camera, capture time and JPEG quality. Camera and JPEG
quality are recorded because they can differ by class, which a model could exploit.

    uv run python -m src.inventory      # both datasets -> data/inventory.csv, data/paddy_inventory.csv
"""

from __future__ import annotations

import argparse
import hashlib
import io
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from PIL.ExifTags import IFD, Base

from src.config import REPO_ROOT, dataset_dir, load_config

COLUMNS = ["photo_id", "folder", "file_bytes", "sha256", "width", "height", "format", "mode",
           "exif_orientation", "camera", "taken_at", "jpeg_quality", "jpeg_quality_exact", "phash",
           "opened_ok", "error"]
INTEGER_COLUMNS = ["file_bytes", "width", "height", "exif_orientation", "jpeg_quality"]

# Luminance quantisation table of the JPEG standard (Annex K), natural order: the IJG reference.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
])
_IJG_TABLES = {  # the luminance table the IJG library writes for each quality 1..100
    q: np.clip((_STD_LUMA * (5000 // q if q < 50 else 200 - 2 * q) + 50) // 100, 1, 255)
    for q in range(1, 101)
}


def jpeg_quality(image: Image.Image) -> tuple[int | None, bool | None]:
    """(closest IJG quality 1-100, whether the table matches it exactly); (None, None) if not a JPEG.

    Only the luminance table is compared, so "exact" means the table equals the IJG one for that
    quality, not that the IJG library wrote the file.
    """
    tables = getattr(image, "quantization", None)
    if not tables or 0 not in tables:
        return None, None
    luma = np.asarray(tables[0])  # Pillow returns tables in natural order, like _STD_LUMA
    errors = {q: int(np.abs(luma - table).sum()) for q, table in _IJG_TABLES.items()}
    best = min(errors, key=errors.get)
    return best, errors[best] == 0


def load_for_hash(source: str | Path | BinaryIO, draft_size: int) -> Image.Image:
    """The exact decode used for perceptual hashes; Stage 2 must hash the same way.

    JPEGs are decoded at a reduced scale (up to 8x smaller per side), which moves a 64-bit
    hash by 0-4 bits compared with a full decode.
    """
    with Image.open(source) as image:
        image.draft("RGB", (draft_size, draft_size))  # must come before decoding
        return ImageOps.exif_transpose(image).convert("RGB")


def perceptual_hash(image: Image.Image, hash_size: int) -> str:
    return str(imagehash.phash(image, hash_size=hash_size))


def describe_file(path: Path, root: Path, hash_size: int, draft_size: int) -> dict[str, Any]:
    """Everything recorded about one file. Never raises: failures are recorded in ``error``."""
    row: dict[str, Any] = {"photo_id": path.relative_to(root).as_posix(), "opened_ok": False}
    try:
        data = path.read_bytes()  # read once; both decodes below work from memory
        row.update(file_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        with Image.open(io.BytesIO(data)) as image:
            exif = image.getexif()
            orientation = exif.get(Base.Orientation, 1)
            if not isinstance(orientation, int):  # malformed tag: treat as upright
                orientation = 1
            quality, exact = jpeg_quality(image)
            # Record the size as the pixels are used (rotation tag applied), like load_for_hash.
            upright = image.size[::-1] if orientation in (5, 6, 7, 8) else image.size
            camera = " ".join(str(exif.get(t, "")).strip() for t in (Base.Make, Base.Model)).strip()
            row.update({
                "width": upright[0], "height": upright[1], "format": image.format, "mode": image.mode,
                "exif_orientation": orientation,
                "camera": camera or None,
                "taken_at": exif.get_ifd(IFD.Exif).get(Base.DateTimeOriginal),
                "jpeg_quality": quality, "jpeg_quality_exact": exact,
            })
            image.load()  # forces a full decode: truncated or corrupt files fail here
        row["phash"] = perceptual_hash(load_for_hash(io.BytesIO(data), draft_size), hash_size)
        row["opened_ok"] = True
    except Exception as exc:  # recorded in the inventory and reported, never silently skipped
        # Drop object addresses ("<... at 0x7f...>") so the recorded text is the same on every run.
        row["error"] = re.sub(r" ?<[^<>]* at 0x[0-9a-f]+>", "", f"{type(exc).__name__}: {exc}")
    return row


def scan(root: Path, hash_size: int, draft_size: int, workers: int | None = None,
         exclude: set[str] | None = None) -> pd.DataFrame:
    """Describe every file under ``root`` except the IDs in ``exclude``, in ``photo_id`` order."""
    exclude = exclude or set()
    ids = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    files = [root / i for i in ids if i not in exclude]
    describe = partial(describe_file, root=root, hash_size=hash_size, draft_size=draft_size)
    if workers == 1:
        rows = [describe(p) for p in files]
    else:
        # spawn, not fork: forking a process that already runs threads (torch, Jupyter) can deadlock
        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as pool:
            rows = list(pool.map(describe, files, chunksize=16))
    frame = pd.DataFrame(rows, columns=COLUMNS)  # fixed columns, whatever the first file was
    frame["folder"] = frame["photo_id"].astype("string").str.extract(r"^([^/]+)/", expand=False)
    return frame.astype({c: "Int64" for c in INTEGER_COLUMNS})


def read_inventory(path: str | Path) -> pd.DataFrame:
    """Read an inventory CSV back with integer columns as integers (even where some are missing)."""
    return pd.read_csv(REPO_ROOT / path, dtype={c: "Int64" for c in INTEGER_COLUMNS})


def dataset_inventory(config: dict[str, Any], name: str, workers: int | None = None) -> pd.DataFrame:
    """Inventory of dataset ``name`` (base.yaml); ``class`` is set for folders mapped to our classes."""
    spec, hashing = config["datasets"][name], config["hashing"]
    root = dataset_dir(config, name)
    metadata = spec.get("metadata_file")
    frame = scan(root, hashing["hash_size"], hashing["draft_size"], workers,
                 exclude={metadata} if metadata else None)
    to_class = {folder: label for label, folder in spec["class_folders"].items()}
    frame.insert(2, "class", frame["folder"].map(to_class))
    if metadata:  # Paddy Doctor: join variety and age, and require a one-to-one match with the files
        meta = pd.read_csv(root / metadata)
        meta["photo_id"] = meta["label"] + "/" + meta["image_id"]
        frame = frame.merge(meta[["photo_id", "variety", "age"]], on="photo_id", how="outer",
                            indicator=True, validate="1:1")
        unmatched = frame.loc[frame["_merge"] != "both", ["photo_id", "_merge"]]
        if len(unmatched):
            raise ValueError(f"{metadata} and the files disagree: {unmatched.head().to_dict('records')}")
        frame = frame.drop(columns="_merge")
    return frame


def write_inventories(config: dict[str, Any], workers: int | None = None) -> dict[str, pd.DataFrame]:
    """Build and write the inventory of every dataset in ``inventory.files``."""
    frames = {}
    for name, relative in config["inventory"]["files"].items():
        frames[name] = dataset_inventory(config, name, workers)
        frames[name].to_csv(REPO_ROOT / relative, index=False)
        print(f"{name}: {len(frames[name]):,} files, {int(frames[name]['opened_ok'].sum()):,} decode fully "
              f"-> {relative}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the inventories of both datasets.")
    parser.add_argument("--config", default="configs/inventory.yaml")
    parser.add_argument("--workers", type=int, help="processes to use (default: all CPU cores)")
    args = parser.parse_args()
    write_inventories(load_config(args.config), args.workers)


if __name__ == "__main__":
    main()
