"""Stage 6: describe every tile (or whole photo) of every primary photo once, and cache the result.

A cache is one settings file in configs/embed/: an encoder, a tile size, a scale (``common``: each photo
first brought to the common leaf scale by ``rescale_factor``; ``native``: as stored), optional Otsu cleaning,
and the number of versions. Version 0 is the clean photo; versions 1-7 change the whole photo first (random
flip and 90-degree rotation, zoom +-15%, colour jitter: brightness, contrast, saturation +-0.2, hue +-0.02)
and are then re-tiled. Each version's randomness is seeded by (seed, version, photo), so every encoder sees
the same augmented photos. Augmented versions are for training photos only; the Stage 7 loaders enforce it.

Output: cache/<name>/v<k>/<photo key>.npz with ``cls`` (n, dim) and, for DINO encoders, ``mean`` (n, dim) in
half precision, plus ``boxes`` (x, y, row, col) in the scaled photo. Whole-photo caches store one row per
size. Resumable: finished files are skipped; each file is written whole, then renamed.

    uv run python -m src.embed --config configs/embed/e1_dinov3_vitb16.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.v2 import functional as F

from src import otsu, tiling
from src.config import REPO_ROOT, dataset_dir, load_config
from src.encoders import describe, load_encoder
from src.inventory import read_inventory
from src.leaf_width import rescale_factor
from src.photo_dataset import prepare_square
from src.records import get_code_version, get_hardware_info
from src.seeding import seed_everything


def photo_key(photo_id: str) -> str:
    return photo_id.replace("/", "__").removesuffix(".jpg")


def version_settings(augment: dict[str, Any], seed: int, version: int, photo_id: str) -> dict[str, Any]:
    """The random whole-photo change of one version of one photo (version 0: none)."""
    if version == 0:
        return {"zoom": 1.0, "flip": False, "rot90": 0, "brightness": 1.0, "contrast": 1.0, "saturation": 1.0,
                "hue": 0.0}
    rng = np.random.default_rng([seed, version, zlib.crc32(photo_id.encode())])
    j = augment["jitter"]
    zoom = augment["zoom"]
    return {"zoom": float(rng.uniform(1 - zoom, 1 + zoom)), "flip": bool(rng.random() < 0.5),
            "rot90": int(rng.integers(4)), "brightness": float(rng.uniform(1 - j, 1 + j)),
            "contrast": float(rng.uniform(1 - j, 1 + j)), "saturation": float(rng.uniform(1 - j, 1 + j)),
            "hue": float(rng.uniform(-augment["hue"], augment["hue"]))}


def load_version(path: str, factor: float, change: dict[str, Any]) -> np.ndarray:
    """The photo scaled by ``factor`` x zoom, with the version's flip, rotation and colour change."""
    with Image.open(path) as image:
        scale = factor * change["zoom"]
        image.draft("RGB", (int(image.width * scale), int(image.height * scale)))  # only shrinks when it can
        image = ImageOps.exif_transpose(image).convert("RGB")
        size = (round(image.width * scale), round(image.height * scale)) if scale != 1 else image.size
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
    x = torch.from_numpy(np.asarray(image)).permute(2, 0, 1)
    if change["flip"]:
        x = torch.flip(x, dims=(2,))
    x = torch.rot90(x, change["rot90"], dims=(1, 2))
    if change["brightness"] != 1.0:
        x = F.adjust_brightness(x, change["brightness"])
        x = F.adjust_contrast(x, change["contrast"])
        x = F.adjust_saturation(x, change["saturation"])
        x = F.adjust_hue(x, change["hue"])
    return x.permute(1, 2, 0).contiguous().numpy()


class Jobs(torch.utils.data.Dataset):
    """One item per (photo, version) still to do: the tiles (or whole-photo images) and their boxes."""

    def __init__(self, config: dict[str, Any], todo: list[tuple[str, int]],
                 factors: dict[str, float]) -> None:
        self.config, self.todo, self.factors = config, todo, factors
        self.root = dataset_dir(config, "primary")

    def __len__(self) -> int:
        return len(self.todo)

    def __getitem__(self, i: int) -> tuple[str, int, np.ndarray, np.ndarray]:
        photo_id, version = self.todo[i]
        e = self.config["embed"]
        change = version_settings(self.config["augment"], self.config["seed"], version, photo_id)
        rgb = load_version(str(self.root / photo_id), self.factors[photo_id], change)
        if e["otsu"]:
            rgb = otsu.clean(rgb)
        if e.get("whole_sizes"):
            images = [prepare_square(rgb, s) for s in e["whole_sizes"]]
            return photo_id, version, images, np.zeros((len(images), 4), np.int32)
        tiles, boxes = tiling.cut(rgb, e["tile"])
        if e["input"] != e["tile"]:  # e.g. 448-px tiles shrunk to 224 (the big-tile ablation)
            tiles = np.stack([np.asarray(Image.fromarray(t).resize((e["input"],) * 2, Image.Resampling.BOX))
                              for t in tiles])
        return photo_id, version, tiles, boxes


def scale_factors(config: dict[str, Any], photo_ids: list[str]) -> dict[str, float]:
    if config["embed"]["scale"] == "native":
        return dict.fromkeys(photo_ids, 1.0)
    inventory = read_inventory(config["inventory"]["files"]["primary"]).set_index("photo_id")
    decision = json.loads((REPO_ROOT / config["pixel_scale"]["decision_file"]).read_text())
    short = inventory.loc[photo_ids, ["width", "height"]].min(axis=1)
    return {p: float(rescale_factor(s, decision)) for p, s in short.items()}


def save(target, cls: torch.Tensor, mean: torch.Tensor | None, boxes: np.ndarray) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"cls": cls.half().numpy(), "boxes": boxes}
    if mean is not None:
        arrays["mean"] = mean.half().numpy()
    partial = target.with_name(target.stem + ".tmp.npz")
    np.savez(partial, **arrays)
    os.replace(partial, target)


def run(config: dict[str, Any], workers: int = 6) -> int:
    from tqdm import tqdm

    e = config["embed"]
    seed_everything(config["seed"], config["determinism"])
    inventory = read_inventory(config["inventory"]["files"]["primary"])
    photo_ids = sorted(inventory.loc[inventory["opened_ok"], "photo_id"])
    folder = REPO_ROOT / "cache" / e["name"]
    todo = [(p, v) for v in range(e["versions"]) for p in photo_ids
            if not (folder / f"v{v}" / f"{photo_key(p)}.npz").exists()]
    if not todo:
        return 0
    meta = {"config": config, "code_version": get_code_version(), "hardware": get_hardware_info(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "to_do": len(todo)}
    device = torch.device("cuda")
    model = load_encoder(config, e["encoder"], device)
    spec = config["encoders"][e["encoder"]]
    jobs = Jobs(config, todo, scale_factors(config, photo_ids))
    loader = torch.utils.data.DataLoader(jobs, batch_size=None, num_workers=workers, prefetch_factor=4)
    for photo_id, version, images, boxes in tqdm(loader, total=len(todo), desc=e["name"]):
        with torch.autocast("cuda", dtype=torch.float16):
            if e.get("whole_sizes"):  # one image per size, each described on its own
                outputs = [describe(model, image[None], spec, config, device) for image in images]
            else:
                outputs = [describe(model, images[i:i + e["batch"]], spec, config, device)
                           for i in range(0, len(images), e["batch"])]
        cls = torch.cat([o[0] for o in outputs]).float()
        mean = None if outputs[0][1] is None else torch.cat([o[1] for o in outputs]).float()
        save(folder / f"v{version}" / f"{photo_key(photo_id)}.npz", cls, mean, np.asarray(boxes))
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (folder / f"record_{meta['started'].replace(':', '')}.json").write_text(json.dumps(meta, indent=2,
                                                                                       default=str))
    return len(todo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"{config['embed']['name']}: {run(config, args.workers)} photo-versions described")


if __name__ == "__main__":
    main()
