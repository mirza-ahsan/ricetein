"""Stage 1 pixel-scale check: are smaller photos crops (same zoom) or resized (zoomed out)?

The rule is fixed in the pre-declared methodology: on 30 photos drawn across the size range, measure
leaf width in pixels; if its Spearman correlation with the photo's shorter side is at least 0.5,
every photo is rescaled to the median leaf width before tiling, otherwise native resolution is kept.

Measuring protocol (declared before anyone measures): on the main leaf (the most central one in
focus), click its two edges straight across the blade at its widest visible point, perpendicular
to the midrib. Each measurer works alone and does not look at the others' numbers.

    uv run python -m src.leaf_width measure --measurer <name>   # opens a window per photo
    uv run python -m src.leaf_width decide                      # applies the rule
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import REPO_ROOT, dataset_dir, load_config
from src.inventory import read_inventory

COLUMNS = ["dataset", "photo_id", "measurer", "x1", "y1", "x2", "y2", "width_px", "measured_at"]


def select_photos(inventory: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """One random photo from each of ``n`` equal-width bands of the shorter side (log scale, from the
    smallest to the largest photo), so the whole size range is covered evenly. An empty band takes
    the unpicked photo nearest its centre. Seeded and independent of row order."""
    photos = inventory[inventory["opened_ok"]].assign(short_side=lambda d: d[["width", "height"]].min(axis=1))
    photos = photos.sort_values(["short_side", "photo_id"], ignore_index=True)
    log_side = np.log(photos["short_side"].to_numpy(dtype=float))
    edges = np.linspace(log_side.min(), log_side.max(), n + 1)
    band = np.clip(np.searchsorted(edges, log_side, side="right") - 1, 0, n - 1)
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for b in range(n):
        members = np.flatnonzero(band == b)
        if len(members) == 0:
            centre = (edges[b] + edges[b + 1]) / 2
            members = np.array([i for i in np.argsort(np.abs(log_side - centre)) if i not in picks][:1])
        picks.append(int(rng.choice(members)))
    return photos.loc[picks, ["photo_id", "class", "width", "height", "short_side"]].reset_index(drop=True)


def load_measurements(path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


def measure(config: dict[str, Any], measurer: str) -> None:
    """Show each selected photo; the measurer clicks the two leaf edges. Progress is saved per photo."""
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    settings = config["pixel_scale"]
    target = REPO_ROOT / settings["measurements_file"]
    inventory = read_inventory(config["inventory"]["files"]["primary"])
    selected = select_photos(inventory, settings["n_photos"], config["seed"])
    done = set(load_measurements(target).query("measurer == @measurer")["photo_id"])
    todo = selected[~selected["photo_id"].isin(done)]
    print(f"{measurer}: {len(done)} done, {len(todo)} to go. Left-click both edges; right-click undoes; "
          "close the window to stop (progress is kept).")
    root = dataset_dir(config, "primary")
    for number, photo in enumerate(todo.itertuples(), start=len(done) + 1):
        with Image.open(root / photo.photo_id) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.set_title(f"{number}/{len(selected)}  {photo.photo_id}\nclick both edges of the main leaf, "
                     "across its widest point, perpendicular to the midrib", fontsize=10)
        ax.set_axis_off()
        clicks = fig.ginput(2, timeout=0)
        plt.close(fig)
        if len(clicks) < 2:
            print("Stopped; progress saved.")
            return
        (x1, y1), (x2, y2) = clicks
        row = {"dataset": "primary", "photo_id": photo.photo_id, "measurer": measurer,
               "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
               "width_px": round(float(np.hypot(x2 - x1, y2 - y1)), 1),
               "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        pd.DataFrame([row], columns=COLUMNS).to_csv(target, mode="a", header=not target.exists(), index=False)
    print("All photos measured.")


def decide(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-declared rule; ``complete`` is False until every measurer has done every photo."""
    settings = config["pixel_scale"]
    inventory = read_inventory(config["inventory"]["files"]["primary"])
    selected = select_photos(inventory, settings["n_photos"], config["seed"])
    data = load_measurements(REPO_ROOT / settings["measurements_file"])
    data = data[(data["dataset"] == "primary") & data["photo_id"].isin(selected["photo_id"])]
    per_measurer = data.pivot_table(index="photo_id", columns="measurer", values="width_px", aggfunc="last")
    result: dict[str, Any] = {"selected": selected, "measurers": list(per_measurer.columns),
                              "n_measured": int(per_measurer.notna().all(axis=1).sum())}
    complete = (len(per_measurer.columns) >= settings["measurers"]
                and result["n_measured"] == len(selected))
    result["complete"] = complete
    if not complete:
        return result
    table = selected.set_index("photo_id").join(per_measurer)
    widths = table[result["measurers"]]
    table["leaf_width_px"] = widths.mean(axis=1)
    rho, p_value = spearmanr(table["leaf_width_px"], table["short_side"])
    between = widths.iloc[:, :2]  # agreement of the first two measurers
    relative_gap = (between.iloc[:, 0] - between.iloc[:, 1]).abs() / table["leaf_width_px"]
    result.update({
        "table": table.reset_index(),
        "spearman_rho": float(rho), "p_value": float(p_value),
        "rescale": bool(rho >= settings["spearman_threshold"]),
        "median_leaf_width_px": float(table["leaf_width_px"].median()),
        "agreement_spearman": float(spearmanr(between.iloc[:, 0], between.iloc[:, 1])[0]),
        "median_relative_gap": float(relative_gap.median()),
        "max_relative_gap": float(relative_gap.max()),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["select", "measure", "decide"])
    parser.add_argument("--measurer", help="your first name (lower case), for 'measure'")
    parser.add_argument("--config", default="configs/inventory.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "select":
        print(select_photos(read_inventory(config["inventory"]["files"]["primary"]),
                            config["pixel_scale"]["n_photos"], config["seed"]).to_string())
    elif args.action == "measure":
        if not args.measurer:
            parser.error("--measurer is required")
        measure(config, args.measurer.strip().lower())
    else:
        result = decide(config)
        if not result["complete"]:
            print(f"Not complete: {result['n_measured']}/{len(result['selected'])} photos measured by "
                  f"all of {result['measurers']} (need {config['pixel_scale']['measurers']} measurers).")
            return
        verdict = "RESCALE to the median leaf width" if result["rescale"] else "keep NATIVE resolution"
        print(f"Spearman rho = {result['spearman_rho']:.3f} (p = {result['p_value']:.3g}) -> {verdict}")


if __name__ == "__main__":
    main()
