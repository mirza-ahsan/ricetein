"""Stage 1 pixel-scale check: are smaller photos crops (same zoom) or resized (zoomed out)?

The rule is fixed in the pre-declared methodology: on 30 photos drawn across the size range, measure
leaf width in pixels; if its Spearman correlation with the photo's shorter side is at least 0.5,
every photo is rescaled to the median leaf width before tiling, otherwise native resolution is kept.

Measuring protocol (declared before anyone measures):
- Measure the main leaf: the most central leaf that is in focus.
- Click its two edges (the blade margins, not lesion borders) straight across the blade at its
  widest visible point, perpendicular to the midrib.
- If no leaf can be measured that way (blade runs off the frame across its width, leaf folded or
  seen edge-on, nothing in focus, two leaves equally central), press U to mark it unmeasurable.
- A photo counts only if every measurer measured it; photos anyone marked unmeasurable are left out
  of the correlation, and how many were left out is reported.
- Each measurer works alone, in their own shuffled order, into their own file, and does not look at
  the others' numbers until both are done.

    uv run python -m src.leaf_width select                      # once: freeze the 30 photos (committed)
    uv run python -m src.leaf_width measure --measurer <name>   # each measurer, on their own
    uv run python -m src.leaf_width decide                      # applies the rule, writes the decision

Coordinates are in full-resolution pixels with matplotlib's pixel-centre convention (0.5 off PIL's
corner convention); widths use differences only, so they are unaffected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import REPO_ROOT, dataset_dir, load_config
from src.inventory import read_inventory

DATASET = "primary"  # Stage 10 adds Paddy Doctor measurements for the zoom calibration
COLUMNS = ["dataset", "photo_id", "measurer", "x1", "y1", "x2", "y2", "width_px", "status", "measured_at"]
INSTRUCTIONS = (
    "Main leaf (most central, in focus): click both blade edges across its widest visible point, "
    "perpendicular to the midrib.\n"
    "Enter = accept (2 points)   Backspace = undo   U = unmeasurable   Esc / close = stop (progress kept)   "
    "Zoom with the toolbar, then switch it off to click."
)


# ---------------------------------------------------------------- selection
def select_photos(inventory: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """One random photo from each of ``n`` equal-width bands of the shorter side (log scale, from the
    smallest to the largest photo), so the whole size range is covered evenly. An empty band takes
    the nearest photo not yet picked. Seeded and independent of row order; never repeats a photo."""
    photos = inventory[inventory["opened_ok"]].assign(short_side=lambda d: d[["width", "height"]].min(axis=1))
    photos = photos.sort_values(["short_side", "photo_id"], ignore_index=True)
    if n > len(photos):
        raise ValueError(f"cannot pick {n} photos from {len(photos)}")
    log_side = np.log(photos["short_side"].to_numpy(dtype=float))
    edges = np.linspace(log_side.min(), log_side.max(), n + 1)
    band = np.minimum(np.searchsorted(edges, log_side, side="right") - 1, n - 1)
    rng = np.random.default_rng(seed)
    taken = np.zeros(len(photos), dtype=bool)
    for b in range(n):
        members = np.flatnonzero((band == b) & ~taken)
        if len(members):
            pick = int(rng.choice(members))
        else:
            centre = (edges[b] + edges[b + 1]) / 2
            pick = int(next(i for i in np.argsort(np.abs(log_side - centre), kind="stable") if not taken[i]))
        taken[pick] = True
    chosen = photos[taken]  # already in size order
    return chosen[["photo_id", "class", "width", "height", "short_side"]].reset_index(drop=True)


def selection(config: dict[str, Any]) -> pd.DataFrame:
    """The frozen selection (written once by ``select`` and committed)."""
    path = REPO_ROOT / config["pixel_scale"]["selection_file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m src.leaf_width select` once and commit it")
    return pd.read_csv(path, dtype={"photo_id": str})


# ---------------------------------------------------------------- measuring
def measurer_file(config: dict[str, Any], measurer: str) -> Path:
    return REPO_ROOT / config["pixel_scale"]["measurements_dir"] / f"{measurer}.csv"


def read_measurements(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(path, dtype={"dataset": str, "photo_id": str, "measurer": str, "status": str})
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"{path} has columns {list(frame.columns)}, expected {COLUMNS}")
    return frame


def click_leaf_edges(image: Any, title: str) -> tuple[str, list[tuple[float, float]]]:
    """Show ``image``; return ("accept", [p1, p2]), ("unmeasurable", []) or ("stop", [])."""
    import matplotlib.pyplot as plt

    keymaps = {k: [] for k in plt.rcParams if k.startswith("keymap.")}  # only our keys do anything
    with plt.rc_context(keymaps):
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.subplots_adjust(0, 0, 1, 0.92)
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)
        points: list[tuple[float, float]] = []
        drawn: list[Any] = []
        outcome = {"action": "stop"}

        def redraw() -> None:
            for artist in drawn:
                artist.remove()
            drawn.clear()
            if points:
                xs, ys = zip(*points)
                drawn.extend(ax.plot(xs, ys, "r+-", markersize=16, linewidth=1))
            fig.canvas.draw_idle()

        def finish(action: str) -> None:
            outcome["action"] = action
            fig.canvas.stop_event_loop()

        def on_click(event: Any) -> None:
            toolbar = getattr(fig.canvas, "toolbar", None)
            if event.inaxes is not ax or event.button != 1 or (toolbar is not None and toolbar.mode):
                return  # zooming, panning or clicking outside the plot is never a measurement
            height, width = np.asarray(image).shape[:2]
            if not (-0.5 <= event.xdata <= width - 0.5 and -0.5 <= event.ydata <= height - 0.5):
                return  # the blank margin around the photo is inside the plot but not on the photo
            if len(points) < 2:
                points.append((float(event.xdata), float(event.ydata)))
                redraw()

        def on_key(event: Any) -> None:
            if event.key == "backspace" and points:
                points.pop()
                redraw()
            elif event.key == "enter" and len(points) == 2:
                finish("accept")
            elif event.key in ("u", "U"):
                finish("unmeasurable")
            elif event.key == "escape":
                finish("stop")

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", lambda event: finish("stop"))
        plt.show(block=False)
        fig.canvas.start_event_loop(timeout=-1)
        plt.close(fig)
    return outcome["action"], points if outcome["action"] == "accept" else []


Asker = Callable[[Any, str], tuple[str, list]]


def measure(config: dict[str, Any], measurer: str, ask: Asker = click_leaf_edges) -> None:
    """Show each selected photo (in this measurer's own shuffled order) and save every answer at once."""
    from PIL import Image, ImageOps

    selected = selection(config)
    target = measurer_file(config, measurer)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = read_measurements(target)
    mine = (rows["dataset"] == DATASET) & rows["photo_id"].isin(selected["photo_id"])
    done = set(rows.loc[mine, "photo_id"])
    # A shuffled order per measurer, so drift during a session cannot line up with photo size.
    name_seed = int.from_bytes(hashlib.sha256(measurer.encode()).digest()[:4], "big")
    order = np.random.default_rng([config["seed"], name_seed]).permutation(len(selected))
    todo = [p for p in selected["photo_id"].iloc[order] if p not in done]
    print(f"{measurer}: {len(done)} of {len(selected)} done.\n{INSTRUCTIONS}")
    root = dataset_dir(config, DATASET)
    for number, photo_id in enumerate(todo, start=len(done) + 1):
        with Image.open(root / photo_id) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
        action, points = ask(image, f"{number}/{len(selected)}   {photo_id}\n{INSTRUCTIONS}")
        if action == "stop":
            print("Stopped; progress saved.")
            return
        row = {"dataset": DATASET, "photo_id": photo_id, "measurer": measurer, "status": action,
               "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if action == "accept":
            (x1, y1), (x2, y2) = points
            row.update(x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
                       width_px=round(float(np.hypot(x2 - x1, y2 - y1)), 1))
        rows = pd.concat([rows, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
        rows.to_csv(target, index=False)  # rewritten whole after every photo: nothing is lost on a crash
    print("All photos measured.")


# ---------------------------------------------------------------- decision
def decide(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-declared rule once every measurer has answered every photo.

    Returns ``complete=False`` (with progress) until then. When complete it also writes the combined
    measurements file and the decision record that Stage 6 reads.
    """
    settings = config["pixel_scale"]
    selected = selection(config)
    folder = REPO_ROOT / settings["measurements_dir"]
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    data = pd.concat([read_measurements(f) for f in files], ignore_index=True) if files \
        else pd.DataFrame(columns=COLUMNS)
    data = data[(data["dataset"] == DATASET) & data["photo_id"].isin(selected["photo_id"])]
    measurers = sorted(data["measurer"].unique())
    strangers = [m for m in measurers if m not in settings["team"]]
    if strangers or len(measurers) > settings["measurers"]:
        raise ValueError(f"measurers {measurers}: expected {settings['measurers']} of {settings['team']}")
    if data.duplicated(["photo_id", "measurer"]).any():
        raise ValueError("a photo was answered twice by the same measurer; fix the measurer's file")
    answered = data.groupby("measurer")["photo_id"].nunique().to_dict()
    result: dict[str, Any] = {"selected": selected, "measurers": measurers, "answered": answered,
                              "complete": len(measurers) == settings["measurers"]
                              and all(n == len(selected) for n in answered.values())}
    if not result["complete"]:
        return result

    status = data.pivot(index="photo_id", columns="measurer", values="status")
    widths = data.pivot(index="photo_id", columns="measurer", values="width_px").astype(float)
    usable = (status == "accept").all(axis=1)
    table = selected.set_index("photo_id").join(widths)
    table["leaf_width_px"] = widths.mean(axis=1)
    table["used"] = usable.reindex(table.index)
    used = table[table["used"]]
    if used["leaf_width_px"].nunique() < 2 or used["short_side"].nunique() < 2:
        raise ValueError("leaf widths or photo sizes are constant: the correlation is undefined")
    rho, p_value = spearmanr(used["leaf_width_px"], used["short_side"])
    first, second = (used[m] for m in measurers)
    gap = (first - second).abs() / used["leaf_width_px"]
    decision = {
        "rule": f"Spearman rho(leaf width, shorter side) >= {settings['spearman_threshold']} -> rescale",
        "spearman_rho": round(float(rho), 4), "p_value": float(p_value),
        "n_used": int(len(used)), "n_unmeasurable": int((~table["used"]).sum()),
        "rescale": bool(rho >= settings["spearman_threshold"]),
        "median_leaf_width_px": round(float(used["leaf_width_px"].median()), 1),
        "measurers": measurers,
        "agreement_spearman": round(float(spearmanr(first, second)[0]), 4),
        "median_relative_gap": round(float(gap.median()), 4), "max_relative_gap": round(float(gap.max()), 4),
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    data.sort_values(["photo_id", "measurer"]).to_csv(REPO_ROOT / settings["measurements_file"], index=False)
    (REPO_ROOT / settings["decision_file"]).write_text(json.dumps(decision, indent=2) + "\n")
    result.update(decision, table=table.reset_index())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["select", "measure", "decide"])
    parser.add_argument("--measurer", help="your first name in lower case, e.g. ahsan (for 'measure')")
    parser.add_argument("--config", default="configs/inventory.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["pixel_scale"]
    if args.action == "select":
        target = REPO_ROOT / settings["selection_file"]
        if target.exists():
            parser.error(f"{target} exists: the selection is frozen (delete it only as a logged decision)")
        chosen = select_photos(read_inventory(config["inventory"]["files"][DATASET]), settings["n_photos"],
                               config["seed"])
        chosen.to_csv(target, index=False)
        print(chosen.to_string(), f"\nwrote {target.relative_to(REPO_ROOT)}")
    elif args.action == "measure":
        name = (args.measurer or "").strip().lower()
        if name not in settings["team"]:
            parser.error(f"--measurer must be one of {settings['team']}")
        measure(config, name)
    else:
        result = decide(config)
        if not result["complete"]:
            print(f"Not complete: answered so far {result['answered'] or 'nothing'} "
                  f"(need {settings['measurers']} measurers x {len(result['selected'])} photos).")
            return
        verdict = "RESCALE to the median leaf width" if result["rescale"] else "keep NATIVE resolution"
        print(f"Spearman rho = {result['spearman_rho']:.3f} (p = {result['p_value']:.3g}, "
              f"n = {result['n_used']}, {result['n_unmeasurable']} unmeasurable) -> {verdict}\n"
              f"median leaf width {result['median_leaf_width_px']:.0f} px; measurers agree with Spearman "
              f"{result['agreement_spearman']:.2f}, median gap {result['median_relative_gap']:.1%}\n"
              f"wrote {settings['measurements_file']} and {settings['decision_file']}")


if __name__ == "__main__":
    main()
