"""Whole photos by split role, for the fine-tuned comparison models (Stage 5a).

Each photo is padded to square (black), optionally Otsu-cleaned, and resized once with area interpolation to
floor(8s/7) px (146, 256 or 512 for s = 128, 224, 448); the result is cached as one git-ignored uint8 array.
``PhotoDataset`` serves one role of one round. Only the training role is ever augmented: asking for
augmentation on validation or test photos raises (the leakage rule, enforced in code and tested).
"""

from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import Pool
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from src import augment, otsu
from src.config import REPO_ROOT, dataset_dir, output_root
from src.inventory import read_inventory
from src.review_tool import load_photo
from src.split import round_column


def cache_size(size: int) -> int:
    return 8 * size // 7


def pad_square(rgb: np.ndarray) -> np.ndarray:
    """The image centred on a black square (non-square photos keep their shape)."""
    h, w = rgb.shape[:2]
    square = np.zeros((max(h, w), max(h, w), 3), np.uint8)
    square[(max(h, w) - h) // 2:(max(h, w) - h) // 2 + h, (max(h, w) - w) // 2:(max(h, w) - w) // 2 + w] = rgb
    return square


def prepare_square(rgb: np.ndarray, side: int) -> np.ndarray:
    """Padded to square and area-resized to ``side`` (whole-photo descriptions, Stage 6)."""
    return cv2.resize(pad_square(rgb), (side, side), interpolation=cv2.INTER_AREA)


def prepare(path: str, side: int, use_otsu: bool) -> np.ndarray:
    """One photo: decoded at reduced scale, padded to square, optionally Otsu-cleaned, area-resized."""
    square = pad_square(np.asarray(load_photo(path, max(1024, side))))
    if use_otsu:
        square = otsu.clean(square)
    return cv2.resize(square, (side, side), interpolation=cv2.INTER_AREA)


def _prepare(job: tuple[str, int, bool]) -> np.ndarray:
    return prepare(*job)


def photo_ids(config: dict[str, Any]) -> list[str]:
    """Every primary photo that opened, sorted: the rows of each image cache."""
    inventory = read_inventory(config["inventory"]["files"]["primary"])
    return sorted(inventory.loc[inventory["opened_ok"], "photo_id"])


def photo_cache(config: dict[str, Any], size: int, use_otsu: bool, workers: int | None = None) -> np.ndarray:
    """(n, c, c, 3) uint8 memory-mapped cache of every photo (``photo_ids`` order, c = cache_size(size));
    built once."""
    photo_ids_ = photo_ids(config)
    side = cache_size(size)
    folder = output_root() / config["wholeimage"]["cache_dir"]
    stem = f"photos_{side}{'_otsu' if use_otsu else ''}"
    data, ids = folder / f"{stem}.u8", folder / f"{stem}.ids.txt"
    spec_path = folder / f"{stem}.spec.json"
    source = ["src/photo_dataset.py", "src/otsu.py", config["inventory"]["files"]["primary"]]
    spec = {"format": 1, "side": side, "otsu": use_otsu,
            "source": {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in source}}
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError(f"{data}: preprocessing changed; archive the old cache before rebuilding")
    if data.exists() and not spec_path.exists():
        raise ValueError(f"{data}: legacy cache lacks provenance; archive before rebuilding")
    if not (ids.exists() and ids.read_text().split("\n") == photo_ids_):
        folder.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, indent=2))
        ids.unlink(missing_ok=True)  # a half-written cache must never look complete
        root = dataset_dir(config, "primary")
        temporary = data.with_suffix(".tmp")
        out = np.memmap(temporary, np.uint8, "w+", shape=(len(photo_ids_), side, side, 3))
        jobs = [(str(root / p), side, use_otsu) for p in photo_ids_]
        with Pool(workers) as pool:
            for i, image in enumerate(pool.imap(_prepare, jobs, chunksize=16)):
                out[i] = image
        out.flush()
        del out
        os.replace(temporary, data)
        ids.write_text("\n".join(photo_ids_))
    if data.stat().st_size != len(photo_ids_) * side * side * 3:
        raise ValueError(f"{data}: resized photo cache has an invalid byte count")
    return np.memmap(data, np.uint8, "r", shape=(len(photo_ids_), side, side, 3))


class PhotoDataset(torch.utils.data.Dataset):
    """The photos of one role in one round: (uint8 CHW image, class index, row)."""

    def __init__(self, folds: pd.DataFrame, images: np.ndarray, image_ids: list[str], round_: int, role: str,
                 size: int, classes: list[str], augmented: bool) -> None:
        if augmented and role != "train":
            raise ValueError(f"augmentation is for training photos only, not {role}")
        self.rows = np.flatnonzero((folds[round_column(round_)] == role).to_numpy())
        self.labels = folds["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        self.cache_row = pd.Index(image_ids).get_indexer(folds["photo_id"])  # folds row -> cache row
        if (self.cache_row[self.rows] < 0).any():
            raise ValueError("some photos of this round are not in the image cache")
        self.images, self.role = images, role
        self.transform = augment.training(size) if augmented else augment.evaluation(size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[i]
        image = torch.from_numpy(np.ascontiguousarray(self.images[self.cache_row[row]])).permute(2, 0, 1)
        return self.transform(image), int(self.labels[row]), int(row)
