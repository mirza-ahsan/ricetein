"""Freeze a 50-photo sample and render original/cleaned pairs using Stage 5's actual preprocessing."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.config import REPO_ROOT, dataset_dir, load_config, output_root
from src.inventory import read_inventory
from src.photo_dataset import prepare


def run(config):
    target = REPO_ROOT / "data/otsu_review_sample.csv"
    ids = sorted(read_inventory(config["inventory"]["files"]["primary"]).photo_id)
    rng = np.random.default_rng(config["seed"])
    if target.exists():
        chosen = pd.read_csv(target).photo_id.tolist()
    else:
        chosen = [ids[i] for i in rng.choice(len(ids), 50, replace=False)]
        pd.DataFrame({"photo_id": chosen}).to_csv(target, index=False)
    root = dataset_dir(config, "primary")
    folder = output_root() / "cache/otsu_review"
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(chosen), 8):
        canvas = Image.new("RGB", (2040, 1080), "#444444")
        for i, photo_id in enumerate(chosen[start : start + 8]):
            x, y = i % 4 * 510, i // 4 * 540
            for k, otsu in enumerate((False, True)):
                rgb = prepare(str(root / photo_id), 256, otsu)
                tile = Image.fromarray(rgb).resize((250, 250))
                canvas.paste(tile, (x + k * 255, y))
                ImageDraw.Draw(canvas).text(
                    (x + k * 255 + 3, y + 250),
                    f"{start + i + 1}: {'Otsu' if otsu else 'original'}",
                    fill="white",
                )
        canvas.crop((0, 0, 2040, 810)).save(folder / f"page_{start // 8 + 1:02}.jpg", quality=94)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sanity.yaml")
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
