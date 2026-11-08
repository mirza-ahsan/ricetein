"""Stage 7.1: whole-photo linear probes; standardisation fitted on training descriptions only."""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import metrics
from src.config import cache_folder, load_config
from src.embed import completed_cache, photo_key
from src.records import RunRecorder, get_default_runs_dir, require_clean_code, reusable_run
from src.seeding import seed_everything
from src.split import read_folds, round_column
from src.train_wholeimage import prediction_table


def descriptors(config, folds, versions):
    p = config["probe"]
    folder = cache_folder(p["cache"])
    complete = completed_cache(config, p["cache"])
    if complete["versions"] < versions:
        raise ValueError("whole-photo cache is incomplete")
    spec = json.loads((folder / "spec.json").read_text())
    if spec["embed"]["whole_sizes"] != p["sizes"]:
        raise ValueError("whole-photo cache size order differs from probe configuration")
    index = p["sizes"].index(p["size"])
    rows = []
    for v in range(versions):
        photos = []
        for photo_id in folds.photo_id:
            with np.load(folder / f"v{v}" / f"{photo_key(photo_id)}.npz") as saved:
                photos.append(saved[p["descriptor"]][index].astype(np.float32))
        rows.append(photos)
    return np.asarray(rows)


def fit_round(x, folds, round_, classes, strength, seed):
    role = folds[round_column(round_)]
    train = np.flatnonzero((role == "train").to_numpy())
    val = np.flatnonzero((role == "val").to_numpy())
    labels = folds["class"].map({c: k for k, c in enumerate(classes)}).to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1 / strength, class_weight="balanced", max_iter=5000, random_state=seed),
    )
    model.fit(x[:, train].reshape(-1, x.shape[-1]), np.tile(labels[train], len(x)))
    probs = model.predict_proba(x[0, val])[:, np.argsort(model.classes_)]
    return model, val, probs


def run(config):
    require_clean_code()
    config = {**config, "cache_spec_sha256": completed_cache(config, config["probe"]["cache"])["spec_sha256"]}
    seed_everything(config["seed"], config["determinism"])
    folds, checksum = read_folds(config)
    p, classes = config["probe"], config["classes"]
    x = descriptors(config, folds, p["versions"])
    trials = []
    for strength in p["l2_strengths"]:
        scores = []
        for r in range(1, config["split"]["n_folds"] + 1):
            name = f"stage7a_probe_{p['name']}_l2{strength:g}_r{r}"
            if not reusable_run(
                name,
                {**config, "round": r, "l2_strength": strength},
                checksum,
                ("model.pkl", "metrics.json", "predictions.csv"),
            ):
                rec = RunRecorder(
                    name,
                    config={**config, "round": r, "l2_strength": strength},
                    split_checksum=checksum,
                    overwrite=True,
                    data_version={"primary": config["datasets"]["primary"]["version"]},
                )
                model, rows, probs = fit_round(x, folds, r, classes, strength, config["seed"])
                with (rec.run_dir / "model.pkl").open("wb") as f:
                    pickle.dump(model, f)
                predictions = prediction_table(folds, rows, probs, r, "val", classes)
                rec.save_predictions(predictions)
                rec.save_metrics(
                    {
                        "partition": "val",
                        **metrics.scores(predictions.true_class, predictions.pred_class, classes),
                    }
                )
                rec.finish()
            result = json.loads((get_default_runs_dir() / name / "metrics.json").read_text())
            scores.append(result["macro_f1"])
        trials.append(
            {"l2_strength": strength, "mean_val_macro_f1": float(np.mean(scores)), "rounds": scores}
        )
    best = max(trials, key=lambda trial: trial["mean_val_macro_f1"])
    selection = {
        "name": p["name"],
        "partition": "val",
        "split_checksum": checksum,
        "trials": trials,
        "selected": best,
        "runs": [
            f"stage7a_probe_{p['name']}_l2{best['l2_strength']:g}_r{r}"
            for r in range(1, config["split"]["n_folds"] + 1)
        ],
    }
    (get_default_runs_dir() / f"stage7a_selection_{p['name']}.json").write_text(
        json.dumps(selection, indent=2)
    )
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    print(run(load_config(parser.parse_args().config))["selected"])


if __name__ == "__main__":
    main()
