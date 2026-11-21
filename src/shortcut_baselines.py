"""Two more shortcut controls (declared 2026-11-14), for either split.

- ``time_neighbour``: no image at all. Each photo gets the class of the training photo from the same camera
  that was taken closest in time (the training majority class when there is none). It measures how much of
  a split's test score is reachable just because photos of one leaf, or one batch of leaves, sit on both
  sides of it.
- ``background_strict``: the colour-histogram model on paper pixels only: outside the Otsu leaf mask grown
  by 5% of the shorter side, and with low saturation. ``background_only`` (Stage 4b) keeps leaf pixels
  wherever the Otsu mask fails; this control keeps far fewer, so it separates "the paper looks different"
  from "some leaf leaked in".

A settings file lists which baselines to run (``shortcut.baselines``; the six Stage 4b ones are allowed, so
they can be repeated on the session-blocked split) and a tag for the run names: ``stage4b_<name><tag>``.
Nothing is tuned, so test scores are reported directly, as in Stage 4b.

    uv run python -m src.shortcut_baselines --config configs/shortcut.yaml
    uv run python -m src.shortcut_baselines --config configs/session/shortcut.yaml
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src import otsu
from src import sanity_baselines as sanity
from src.config import dataset_dir, load_config, output_root
from src.records import RunRecorder, get_default_runs_dir, require_clean_code, reusable_run
from src.review_tool import load_photo
from src.seeding import seed_everything
from src.session_split import nearest_in_sorted, with_capture
from src.split import read_folds, round_column

MAX_SATURATION = 51   # of 255: paper is nearly colourless, a leaf is not
MARGIN = 0.05         # the leaf mask is grown by this share of the shorter side


def paper_histogram(path: str) -> np.ndarray:
    """HSV histogram (512 values) of the low-saturation pixels well away from the leaf."""
    rgb = np.asarray(load_photo(path, 512))
    size = max(3, round(MARGIN * min(rgb.shape[:2])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    near_leaf = cv2.dilate(otsu.leaf_mask(rgb).astype(np.uint8), kernel).astype(bool)
    pale = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1] < MAX_SATURATION
    return sanity.hsv_histogram(rgb, ~near_leaf & pale)


def paper_histograms(config: dict[str, Any], photo_ids: list[str], workers: int | None = None) -> np.ndarray:
    """(n, 512) paper histograms for ``photo_ids``, cached (git-ignored)."""
    target = output_root() / config["shortcut"]["histogram_cache"]
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if saved["ids"].tolist() == photo_ids:
                return saved["histograms"]
    root = dataset_dir(config, "primary")
    with Pool(workers) as pool:
        values = np.array(pool.map(paper_histogram, [str(root / p) for p in photo_ids], chunksize=16))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, ids=np.array(photo_ids), histograms=values)
    return values


def time_neighbour_round(table: pd.DataFrame, r: int, classes: list[str]) -> pd.DataFrame:
    """Round ``r``: validation and test photos get the class of the nearest-in-time training photo."""
    role = table[round_column(r)]
    train, scored = table[role == "train"], table[role.isin(["val", "test"])]
    predicted = pd.Series(train["class"].value_counts().idxmax(), index=scored.index)
    for camera, photos in scored.groupby("camera"):
        known = train[(train["camera"] == camera) & train["seconds"].notna()].sort_values("seconds")
        photos = photos[photos["seconds"].notna()]
        if not len(known) or not len(photos):
            continue
        times = known["seconds"].to_numpy()
        predicted[photos.index] = known["class"].to_numpy()[nearest_in_sorted(times, photos["seconds"])[0]]
    out = scored[["photo_id", "class"]].rename(columns={"class": "true_class"})
    out.insert(0, "round", r)
    out.insert(1, "partition", role[scored.index].to_numpy())
    out["pred_class"] = predicted.to_numpy()
    for c in classes:
        out[f"prob_{c}"] = (out["pred_class"] == c).astype(float)
    return out


def run(config: dict[str, Any], workers: int | None = None) -> dict[str, Any]:
    require_clean_code()
    seed_everything(config["seed"], config["determinism"])
    settings, classes = config["shortcut"], config["classes"]
    folds, checksum = read_folds(config)
    table = with_capture(folds, config)
    inputs: dict[str, Any] = {}
    if set(settings["baselines"]) & set(sanity.BASELINES[2:]):
        inputs = sanity.features(config, folds, workers)
    if "background_strict" in settings["baselines"]:
        inputs["background_strict"] = paper_histograms(config, folds["photo_id"].tolist(), workers)
    results = {}
    for name in settings["baselines"]:
        run_name = f"stage4b_{name}{settings['tag']}"
        run_config = {**config, "baseline": name}
        if not reusable_run(run_name, run_config, checksum):
            recorder = RunRecorder(run_name, config=run_config, split_checksum=checksum, overwrite=True,
                                   data_version={"primary": config["datasets"]["primary"]["version"]})
            rounds = range(1, config["split"]["n_folds"] + 1)
            if name == "time_neighbour":
                parts = [time_neighbour_round(table, r, classes) for r in rounds]
            else:
                parts = [sanity.predict_round(name, folds, inputs.get(name), r, classes, config["seed"])
                         for r in rounds]
            predictions = pd.concat(parts, ignore_index=True)
            recorder.save_predictions(predictions)
            recorder.save_metrics(sanity.summarise(predictions, config))
            recorder.finish()
        results[name] = json.loads((get_default_runs_dir() / run_name / "metrics.json").read_text())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/shortcut.yaml")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    for name, result in run(load_config(args.config), args.workers).items():
        print(f"{name}: test macro-F1 {100 * result['test']['macro_f1']['mean']:.1f}, "
              f"balanced accuracy {100 * result['test']['balanced_accuracy']['mean']:.1f}")


if __name__ == "__main__":
    main()
