"""Stage 4b: floor scores and shortcut checks on the locked folds (methodology "Sanity ... baselines").

1. majority: always the most frequent training class;  2. random_prior: random guesses in the training
class proportions;  3. colour_only: logistic regression on an 8 x 8 x 8 HSV histogram of the photo;
4. background_only: the same, on the background pixels only (the Otsu leaf mask removed);  5. properties_only:
width, height, file size and tile count (at the common leaf scale);  6. camera_only (declared 2026-11-04):
the EXIF camera model.

Each model is fitted on the training role of every round and predicts that round's validation and test photos
with fixed settings (nothing is tuned). Logistic regressions use balanced class weights, so they are as strong
as a floor reasonably gets on balanced scores. Photos are decoded at 512 px for the histograms (global colour
statistics barely depend on resolution); this cache is git-ignored.

    uv run python -m src.sanity_baselines        # every baseline -> runs/stage4_<name>/ + sanity report
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import metrics, otsu, stats
from src.config import REPO_ROOT, dataset_dir, load_config, path_for
from src.inventory import read_inventory
from src.inventory_report import md_table
from src.leaf_width import rescale_factor
from src.records import RunRecorder
from src.review_tool import load_photo
from src.seeding import seed_everything
from src.split import read_folds, round_column

BASELINES = ("majority", "random_prior", "colour_only", "background_only", "properties_only", "camera_only")


# ---------------------------------------------------------------- features
def hsv_histogram(rgb: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Normalised 8 x 8 x 8 HSV histogram (512 values) of the pixels where ``keep`` is True (all if None)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = None if keep is None else keep.astype(np.uint8)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256]).ravel()
    return hist / hist.sum() if hist.sum() else hist


def photo_histograms(path: str) -> np.ndarray:
    """[colour histogram of the whole photo, histogram of its background pixels] (1,024 values)."""
    rgb = np.asarray(load_photo(path, 512))
    return np.concatenate([hsv_histogram(rgb), hsv_histogram(rgb, ~otsu.leaf_mask(rgb))])


def histograms(config: dict[str, Any], photo_ids: list[str], workers: int | None = None) -> np.ndarray:
    """(n, 1024) histograms for ``photo_ids``, cached (git-ignored) and rebuilt if the photo list changes."""
    target = REPO_ROOT / config["sanity"]["histogram_cache"]
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if saved["ids"].tolist() == photo_ids:
                return saved["histograms"]
    root = dataset_dir(config, "primary")
    with Pool(workers) as pool:
        values = np.array(pool.map(photo_histograms, [str(root / p) for p in photo_ids], chunksize=16))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, ids=np.array(photo_ids), histograms=values)
    return values


def tile_count(width: int, height: int, factor: float, tile: int) -> int:
    return -(-round(width * factor) // tile) * -(-round(height * factor) // tile)


def features(config: dict[str, Any], folds: pd.DataFrame, workers: int | None = None) -> dict[str, Any]:
    """The inputs of each feature baseline, row-aligned with ``folds``."""
    inventory = read_inventory(config["inventory"]["files"]["primary"]).set_index("photo_id")
    inventory = inventory.loc[folds["photo_id"]]
    decision = json.loads((REPO_ROOT / config["pixel_scale"]["decision_file"]).read_text())
    short = inventory[["width", "height"]].min(axis=1)
    factors = short.map(lambda s: rescale_factor(s, decision))
    props = pd.DataFrame({"width": inventory["width"], "height": inventory["height"],
                          "file_bytes": inventory["file_bytes"],
                          "tiles": [tile_count(w, h, f, config["sanity"]["tile"])
                                    for w, h, f in zip(inventory["width"], inventory["height"], factors)]})
    hist = histograms(config, folds["photo_id"].tolist(), workers)
    return {"colour_only": hist[:, :512], "background_only": hist[:, 512:],
            "properties_only": props.to_numpy(float),
            "camera_only": inventory[["camera"]].fillna("unknown").to_numpy()}


def model(name: str, seed: int) -> Any:
    regression = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    if name == "camera_only":
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"), regression)
    return make_pipeline(StandardScaler(), regression)


# ---------------------------------------------------------------- running
def predict_round(name: str, table: pd.DataFrame, inputs: Any, r: int, classes: list[str],
                  seed: int) -> pd.DataFrame:
    """Fit on round r's training photos; probabilities for its validation and test photos."""
    role = table[round_column(r)]
    train, scored = role == "train", role.isin(["val", "test"])
    y = table.loc[train, "class"].to_numpy()
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
    n = int(scored.sum())
    if name == "majority":
        probs = np.tile(np.eye(len(classes))[prior.argmax()], (n, 1))
    elif name == "random_prior":
        rng = np.random.default_rng([seed, r])
        probs = np.eye(len(classes))[rng.choice(len(classes), n, p=prior)]
    else:
        fitted = model(name, seed).fit(inputs[train.to_numpy()], y)
        probs = pd.DataFrame(fitted.predict_proba(inputs[scored.to_numpy()]),
                             columns=fitted.classes_).reindex(columns=classes, fill_value=0).to_numpy()
    out = table.loc[scored, ["photo_id", "class"]].rename(columns={"class": "true_class"})
    out.insert(0, "round", r)
    out.insert(1, "partition", role[scored].to_numpy())
    out["pred_class"] = np.array(classes)[probs.argmax(axis=1)]
    for k, c in enumerate(classes):
        out[f"prob_{c}"] = probs[:, k]
    return out


def summarise(predictions: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Per-round mean +- sd on test (and validation), pooled test scores with bootstrap intervals, ECE."""
    classes, evaluation = config["classes"], config["evaluation"]
    test = predictions[predictions["partition"] == "test"]
    true, pred, boot = test["true_class"], test["pred_class"], evaluation["bootstrap"]
    true_index = true.map({c: i for i, c in enumerate(classes)}).to_numpy()
    probs = metrics.probabilities(test, classes)
    return {"test": metrics.per_round(predictions, classes)["summary"],
            "val": metrics.per_round(predictions, classes, "val")["summary"],
            "pooled_test": metrics.pooled(predictions, classes),
            "pooled_macro_f1": stats.bootstrap(true, pred, stats.macro_f1(classes), boot),
            "pooled_balanced_accuracy": stats.bootstrap(true, pred, stats.balanced_accuracy, boot),
            "ece": stats.ece(probs, true_index, evaluation["calibration_bins"])}


def run(config: dict[str, Any], names: tuple[str, ...] = BASELINES,
        workers: int | None = None) -> dict[str, Any]:
    seed_everything(config["seed"], config["determinism"])
    table, checksum = read_folds(config)
    inputs = features(config, table, workers)
    results = {}
    for name in names:
        recorder = RunRecorder(f"stage4_{name}", config={**config, "baseline": name}, split_checksum=checksum,
                               data_version={"primary": config["datasets"]["primary"]["version"]})
        rounds = range(1, config["split"]["n_folds"] + 1)
        predictions = pd.concat([predict_round(name, table, inputs.get(name), r, config["classes"],
                                               config["seed"]) for r in rounds], ignore_index=True)
        recorder.save_predictions(predictions)
        results[name] = summarise(predictions, config)
        recorder.save_metrics(results[name])
        recorder.finish()
    return results


def report(config: dict[str, Any], results: dict[str, Any]) -> str:
    def pm(s: dict[str, float]) -> str:
        return f"{100 * s['mean']:.1f} ± {100 * s['std']:.1f}"

    def ci(s: dict[str, float]) -> str:
        return f"{100 * s['value']:.1f} [{100 * s['low']:.1f}, {100 * s['high']:.1f}]"

    rows = pd.DataFrame([{"baseline": name, "test macro-F1": pm(r["test"]["macro_f1"]),
                          "test balanced acc.": pm(r["test"]["balanced_accuracy"]),
                          "test accuracy": pm(r["test"]["accuracy"]),
                          "pooled macro-F1 [95% CI]": ci(r["pooled_macro_f1"]),
                          "pooled balanced acc. [95% CI]": ci(r["pooled_balanced_accuracy"]),
                          "hispa recall": f"{100 * r['pooled_test']['per_class']['hispa']['recall']:.1f}",
                          "ECE": f"{r['ece']:.3f}"} for name, r in results.items()])
    text = ("# Stage 4 sanity baselines\n\nBuilt by `src/sanity_baselines.py` from `runs/stage4_*`; do not "
            "edit by hand. Scores in percent on the **test** folds of the locked split (mean ± sd, 5 rounds, "
            "and the pooled out-of-fold predictions with 95% bootstrap intervals). Nothing here is tuned, so "
            "test scores are reported directly. Chance: 25% balanced accuracy.\n\n" + md_table(rows) + "\n")
    (path_for(config, "reports") / "sanity_report.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/sanity.yaml")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    print(report(config, run(config, workers=args.workers)))


if __name__ == "__main__":
    main()
