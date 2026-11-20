"""Stage 10: describe every Paddy Doctor photo once and keep the result (plan: docs/reports/paddy_pre_declaration.md).

For the four tested classes of ``splits/paddy4.csv`` this builds, with the same frozen encoder and the same code
as the Stage 6 caches (clean photo only, no augmentation):

- three tile caches, ``paddy_e1_native``, ``paddy_e1_midway`` and ``paddy_e1_matched``: the photo scaled by the
  declared zoom factor, cut into 224 px tiles and described (same layout as a Stage 6 cache, one version);
- one whole-photo cache, ``paddy_e1_whole``: the photo padded to a square and resized, one description per size;
- the resized, padded photos the whole-image models need (``cache/paddy_photos/``). Otsu cleaning is applied to
  the photo **before** padding, which ``photo_dataset.prepare`` does not do (it pads first; harmless for the
  nearly square primary photos, wrong for 1080 x 1440 photos).

    uv run python -m src.paddy_embed --config configs/paddy_zero_shot.yaml --cache <name>|all|photos

Resumable: a finished photo file is skipped. Nothing here reads a label or a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from src import otsu, tiling
from src.config import REPO_ROOT, cache_folder, dataset_dir, load_config, output_root
from src.embed import load_version, photo_key, save
from src.encoders import describe, load_encoder
from src.photo_dataset import cache_size, pad_square, prepare_square
from src.review_tool import load_photo
from src.records import get_code_version, get_hardware_info, require_clean_code
from src.seeding import seed_everything

TILE, BATCH = 224, 128                      # as configs/embed/tiling.yaml
PHOTO_SETTINGS = [(128, False), (224, False), (224, True), (448, False)]   # (size, Otsu) of the whole-image models
CLEAN = {"zoom": 1.0, "flip": False, "rot90": 0, "brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "hue": 0.0}
SOURCES = ["src/paddy_embed.py", "src/embed.py", "src/encoders.py", "src/tiling.py", "src/otsu.py",
           "src/photo_dataset.py"]


def photo_ids(config: dict[str, Any]) -> list[str]:
    return sorted(pd.read_csv(REPO_ROOT / config["paddy"]["split_file"])["photo_id"])


def description_caches(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Cache name -> its settings: a tile cache with a zoom factor, or the whole-photo cache with its sizes."""
    p = config["paddy"]
    caches = {name: {"kind": "tiles", "zoom": zoom, "factor": p["zoom_factors"][zoom]}
              for zoom, name in p["caches"]["tiles"].items()}
    caches[p["caches"]["whole"]] = {"kind": "whole", "sizes": p["caches"]["whole_sizes"], "factor": 1.0}
    return caches


def cache_spec(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Everything that determines the descriptors; changing it needs a fresh cache."""
    sha = lambda path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()   # noqa: E731
    return {"format": 1, "settings": settings, "tile": TILE, "encoder": config["encoders"][config["paddy"]["encoder"]],
            "normalisation": config["normalisation"], "determinism": config["determinism"],
            "source": {path: sha(path) for path in SOURCES}, "photos": sha(config["paddy"]["split_file"])}


class Jobs(torch.utils.data.Dataset):
    """One item per photo still to do: its tiles and boxes, or its whole-photo images."""

    def __init__(self, config: dict[str, Any], settings: dict[str, Any], todo: list[str]) -> None:
        self.settings, self.todo, self.root = settings, todo, dataset_dir(config, "paddy")

    def __len__(self) -> int:
        return len(self.todo)

    def __getitem__(self, i: int) -> tuple[str, np.ndarray, np.ndarray]:
        photo_id = self.todo[i]
        rgb = load_version(str(self.root / photo_id), self.settings["factor"], CLEAN)
        if self.settings["kind"] == "whole":
            images = [prepare_square(rgb, s) for s in self.settings["sizes"]]   # a list: the sizes differ
            return photo_id, images, np.zeros((len(images), 4), np.int32)
        tiles, boxes = tiling.cut(rgb, TILE)
        return photo_id, tiles, boxes


def validate(config: dict[str, Any], name: str, settings: dict[str, Any]) -> dict[str, Any]:
    folder = cache_folder(name)
    spec = json.loads((folder / "spec.json").read_text())
    if spec != cache_spec(config, settings):
        raise ValueError(f"{folder}: cache settings differ")
    dim = config["encoders"][config["paddy"]["encoder"]]["dim"]
    rows = size = 0
    for photo_id in photo_ids(config):
        path = folder / "v0" / f"{photo_key(photo_id)}.npz"
        with np.load(path, allow_pickle=False) as data:
            n = len(data["cls"])
            ok = n > 0 and data["boxes"].shape == (n, 4)
            for key in ("cls", "mean"):
                ok = ok and data[key].shape == (n, dim) and data[key].dtype == np.float16 and np.isfinite(data[key]).all()
            if not ok or (settings["kind"] == "whole" and n != len(settings["sizes"])):
                raise ValueError(f"{path}: invalid descriptor arrays")
            rows += n
        size += path.stat().st_size
    return {"name": name, "photos": len(photo_ids(config)), "versions": 1, "descriptor_rows": rows, "bytes": size,
            "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()}


def describe_photos(config: dict[str, Any], name: str, workers: int = 6) -> int:
    """Describe the photos of one cache that have no file yet; returns how many were described."""
    from tqdm import tqdm

    require_clean_code()
    settings = description_caches(config)[name]
    seed_everything(config["seed"], config["determinism"])
    folder = cache_folder(name, write=True)
    folder.mkdir(parents=True, exist_ok=True)
    spec, spec_path = cache_spec(config, settings), folder / "spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError(f"{folder}: settings changed; archive this cache before rebuilding")
    elif any(folder.glob("v0/*.npz")):
        raise ValueError(f"{folder}: files without provenance; archive before rebuilding")
    else:
        spec_path.write_text(json.dumps(spec, indent=2))
    todo = [p for p in photo_ids(config) if not (folder / "v0" / f"{photo_key(p)}.npz").exists()]
    if todo:
        device = torch.device("cuda")
        encoder = config["paddy"]["encoder"]
        model = load_encoder(config, encoder, device)
        loader = torch.utils.data.DataLoader(Jobs(config, settings, todo), batch_size=None, num_workers=workers,
                                             prefetch_factor=2 if workers else None)
        for photo_id, images, boxes in tqdm(loader, total=len(todo), desc=name):
            with torch.autocast("cuda", dtype=torch.float16):   # as in src.embed, so the descriptors are comparable
                if settings["kind"] == "whole":
                    outputs = [describe(model, image[None], config["encoders"][encoder], config, device)
                               for image in images]
                else:
                    outputs = [describe(model, images[i:i + BATCH], config["encoders"][encoder], config, device)
                               for i in range(0, len(images), BATCH)]
            save(folder / "v0" / f"{photo_key(photo_id)}.npz", torch.cat([o[0] for o in outputs]).float(),
                 torch.cat([o[1] for o in outputs]).float(), np.asarray(boxes))
        record = {"config": config, "code_version": get_code_version(), "hardware": get_hardware_info(),
                  "described": len(todo)}
        (folder / f"record_{len(list(folder.glob('record_*.json')))}.json").write_text(
            json.dumps(record, indent=2, default=str))
    (folder / "complete.json").write_text(json.dumps(validate(config, name, settings), indent=2))
    return len(todo)


# --------------------------------------------------------------- resized photos for the whole-image models
def prepare_photo(job: tuple[str, int, bool]) -> np.ndarray:
    """One photo for a whole-image model: decoded, Otsu-cleaned **before** padding when asked, padded, resized."""
    path, side, use_otsu = job
    rgb = np.asarray(load_photo(path, max(1024, side)))
    if use_otsu:
        rgb = otsu.clean(rgb)
    return cv2.resize(pad_square(rgb), (side, side), interpolation=cv2.INTER_AREA)


def photo_array(config: dict[str, Any], size: int, use_otsu: bool, workers: int = 4) -> np.ndarray:
    """(n, c, c, 3) uint8 memory map of every Paddy photo in ``photo_ids`` order, c = cache_size(size); built once."""
    ids, side = photo_ids(config), cache_size(size)
    folder = output_root() / "cache" / "paddy_photos"
    stem = f"photos_{side}{'_otsu' if use_otsu else ''}"
    data, listing, spec_path = folder / f"{stem}.u8", folder / f"{stem}.ids.txt", folder / f"{stem}.spec.json"
    spec = {"format": 1, "side": side, "otsu": use_otsu, "otsu_before_padding": True,
            "source": {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
            "photos": hashlib.sha256((REPO_ROOT / config["paddy"]["split_file"]).read_bytes()).hexdigest()}
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError(f"{data}: preprocessing changed; archive the old cache before rebuilding")
    if not (listing.exists() and listing.read_text().split("\n") == ids):
        folder.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, indent=2))
        listing.unlink(missing_ok=True)   # a half-written array must never look complete
        root = dataset_dir(config, "paddy")
        temporary = data.with_suffix(".tmp")
        out = np.memmap(temporary, np.uint8, "w+", shape=(len(ids), side, side, 3))
        with Pool(workers) as pool:
            for i, image in enumerate(pool.imap(prepare_photo, [(str(root / p), side, use_otsu) for p in ids],
                                                chunksize=16)):
                out[i] = image
        out.flush()
        del out
        os.replace(temporary, data)
        listing.write_text("\n".join(ids))
    if data.stat().st_size != len(ids) * side * side * 3:
        raise ValueError(f"{data}: invalid byte count")
    return np.memmap(data, np.uint8, "r", shape=(len(ids), side, side, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/paddy_zero_shot.yaml")
    parser.add_argument("--cache", required=True, help="a cache name, 'all' (the four description caches) or 'photos'")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    config = load_config(args.config)
    caches = description_caches(config)
    if args.cache == "photos":
        for size, use_otsu in PHOTO_SETTINGS:
            print(f"photos {size} otsu {use_otsu}: {len(photo_array(config, size, use_otsu, args.workers))} ready")
        return
    for name in caches if args.cache == "all" else [args.cache]:
        print(f"{name}: {describe_photos(config, name, args.workers)} photos described", flush=True)


if __name__ == "__main__":
    main()
