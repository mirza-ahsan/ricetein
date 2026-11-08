"""Stage 6 counts, storage, deterministic repeated descriptions and blinded tile-neighbour galleries."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from src import embed
from src.config import cache_folder, load_config, output_root
from src.encoders import describe, load_encoder
from src.inventory import read_inventory
from src.inventory_report import md_table
from src.seeding import seed_everything


def check_example(config):
    """Re-describe one clean photo twice, checking repeatability and the saved output. Save a small
    nearest-neighbour gallery across several photos, so a person can inspect lesion/background matches."""
    e = config["embed"]
    inventory = read_inventory(config["inventory"]["files"]["primary"])
    ids = sorted(inventory.loc[inventory.opened_ok, "photo_id"])
    # Label-free fixed sample: no test/validation metric or symptom judgement influences selection.
    chosen = [ids[i] for i in np.linspace(0, len(ids) - 1, 6, dtype=int)]
    jobs = embed.Jobs(config, [(p, 0) for p in chosen], embed.scale_factors(config, chosen))
    seed_everything(config["seed"], config["determinism"])
    device = torch.device("cuda")
    model = load_encoder(config, e["encoder"], device)
    spec = config["encoders"][e["encoder"]]
    _, _, images, _ = jobs[0]

    def encode():
        with torch.autocast("cuda", dtype=torch.float16):
            batches = (
                [im[None] for im in images]
                if e.get("whole_sizes")
                else [images[i : i + e["batch"]] for i in range(0, len(images), e["batch"])]
            )
            outputs = [describe(model, batch, spec, config, device)[0] for batch in batches]
        return torch.cat(outputs).float().numpy()

    first, second = encode(), encode()
    np.testing.assert_array_equal(first, second)
    with np.load(cache_folder(e["name"]) / "v0" / f"{embed.photo_key(chosen[0])}.npz") as saved:
        np.testing.assert_allclose(first, saved["cls"], rtol=1e-3, atol=2e-3)
    model.to("cpu")
    torch.cuda.empty_cache()
    if not e.get("whole_sizes"):
        tiles, features, photo_rows = [], [], []
        for i, photo_id in enumerate(chosen):
            _, _, photo_tiles, _ = jobs[i]
            with np.load(cache_folder(e["name"]) / "v0" / f"{embed.photo_key(photo_id)}.npz") as saved:
                features.extend(saved["cls"].astype(np.float32))
            tiles.extend(photo_tiles)
            photo_rows.extend([i] * len(photo_tiles))
        x = np.asarray(features)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        canvas = Image.new("RGB", (5 * 150, 6 * 170), "white")
        for row, i in enumerate(np.linspace(0, len(x) - 1, 6, dtype=int)):
            similarity = x @ x[i]
            similarity[np.array(photo_rows) == photo_rows[i]] = -np.inf
            neighbours = np.argsort(-similarity)[:4]
            for col, tile_index in enumerate([i, *neighbours]):
                canvas.paste(Image.fromarray(tiles[tile_index]).resize((150, 150)), (col * 150, row * 170))
                ImageDraw.Draw(canvas).text(
                    (col * 150 + 3, row * 170 + 150),
                    "query" if col == 0 else f"{similarity[tile_index]:.3f}",
                    fill="black",
                )
        target = output_root() / "cache/neighbour_galleries"
        target.mkdir(parents=True, exist_ok=True)
        canvas.save(target / f"{e['name']}.jpg")
    return {"repeatable": True, "saved_output_matches": True, "sample_photos": chosen}


def run(config):
    results, checks = [], {}
    for name in config["pipeline"]["caches"]:
        c = load_config(f"configs/embed/{name}.yaml")
        result = embed.validate_cache(c)
        records = [json.loads(p.read_text()) for p in cache_folder(c["embed"]["name"]).glob("record_*.json")]
        result["run_records"] = len(records)
        results.append(result)
        checks[result["name"]] = check_example(c)
    root = output_root()
    manifest = root / "cache/manifest.csv"
    pd.DataFrame(results).to_csv(manifest, index=False)
    target = root / "docs/reports"
    target.mkdir(parents=True, exist_ok=True)
    (target / "cache_checks.json").write_text(json.dumps(checks, indent=2))
    (target / "cache_report.md").write_text(
        "# Stage 6 cache report\n\nAll listed photo/version files were checked for finite half-precision "
        "descriptors and consistent shapes. One clean photo per cache was described twice, with identical "
        "outputs, and checked against its saved descriptors. Label-free nearest-neighbour galleries are "
        "in the ignored `cache/neighbour_galleries/` folder; their visual review is recorded separately.\n\n"
        + md_table(pd.DataFrame(results))
        + "\n"
    )
